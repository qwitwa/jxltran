// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Minimal JXL entropy decoder for jxltran — enough to advance a BitReader
// past a compressed ICC-profile stream so that the byte-aligned frame data
// can be found.  Supports ANS and Brotli-style prefix codes, LZ77, and
// hybrid-uint decoding.

#include "entropy.h"

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include "bits.h"
#include "entropy_trace.h"
#include "frame_header.h"
#include "lfglobal.h"
#include "printf_macros.h"

namespace jxltran {

// ── Internal helper types ────────────────────────────────────────────────────

// Single entry in a Huffman lookup table.
// nbits==255 → extension table; use ext_bits bits to index ext_tables[value].
// nbits==0   → degenerate (single symbol); value is the symbol, consume 0 bits.
struct HuffEntry {
  uint8_t nbits;
  uint8_t ext_bits;
  uint16_t value;
};

// ANS alias-table entry (matches lib/jxl/ans_common.h AliasTable::Entry layout
// for decoding). freq0 is D[i]; freq1_xor_freq0 is D[right_value] ^ D[i].
struct AliasEntry {
  uint16_t symbol;  // right_value (secondary symbol index in this bucket)
  uint16_t cutoff;
  uint32_t offset;
  uint16_t freq0;
  uint16_t freq1_xor_freq0;
};

// One probability distribution (ANS or prefix-code).
struct DistFull {
  bool use_prefix = false;
  int log_alpha = 0;
  int degenerate_sym = -1;  // -1 → not degenerate

  // ANS alias table (size = 1 << log_alpha).
  std::vector<AliasEntry> alias;

  // Prefix lookup table (size = 1 << kPrefixHuffBits) + extension tables.
  std::vector<HuffEntry> huff;
  std::vector<std::vector<HuffEntry>> huff_ext;

  // Hybrid-uint config for this context cluster.
  UintConfig config;
};

// ── CoderState ───────────────────────────────────────────────────────────────

struct CoderState {
  bool use_prefix = false;
  uint32_t ans_state = 0;

  std::vector<uint8_t> clusters;  // [num_dist] → cluster index
  std::vector<DistFull> dists;    // per cluster

  Lz77Params lz77;
  // When LZ77 is enabled: histogram cluster index for distance symbols
  // (libjxl context_map->back() after DecodeHistograms).
  size_t lz_dist_ctx = 0;

  // LZ77 runtime (only allocated when lz77.enabled).
  std::unique_ptr<uint32_t[]> window;
  uint32_t num_to_copy = 0;
  uint32_t copy_pos = 0;
  uint32_t num_decoded = 0;
};

// ── Internal helpers ─────────────────────────────────────────────────────────

namespace {

// Forward declarations (need CoderState to be complete, which it is at this
// point).
static uint32_t DecodeOne(BitReader& br, CoderState& st, size_t ctx);
static bool InitCoderState(BitReader& br, size_t num_dist, bool disallow_lz77,
                           CoderState* st);

static int CeilLog2(int n) {
  if (n <= 1) return 0;
  int b = 0;
  while ((1 << b) < n) ++b;
  return b;
}

// FloorLog2 for n>=1 — matches FloorLog2Nonzero in lib/jxl/base/bits.h (must
// match dec_ans.cc ReadHistogram's unary-coded shift upper bound).
static int FloorLog2Nonzero(int n) {
  assert(n >= 1);
  int b = 0;
  while (n >>= 1) ++b;
  return b;
}

// Reverse low `len` bits of `v` (for LSB-first Huffman table indexing).
static uint32_t RevBits(uint32_t v, int len) {
  uint32_t r = 0;
  for (int i = 0; i < len; ++i) r |= ((v >> i) & 1u) << (len - 1 - i);
  return r;
}

// Reads U8 in [0,255] using 1–11 bits (used in ANS distribution reading).
static int ReadU8(BitReader& br) {
  if (!br.ReadBits(1)) return 0;
  int n = br.ReadBits(3);
  if (n == 0) return 1;
  return static_cast<int>(br.ReadBits(n)) + (1 << n);
}

// Inverse of ReadU8 (libjxl Foley–Rice style U8).
static void WriteU8(BitWriter* bw, int value) {
  if (value <= 0) {
    bw->WriteBits(1, 0);
    return;
  }
  if (value == 1) {
    bw->WriteBits(1, 1);
    bw->WriteBits(3, 0);
    return;
  }
  const int n = FloorLog2Nonzero(static_cast<uint32_t>(value));
  const int u = value - (1 << n);
  bw->WriteBits(1, 1);
  bw->WriteBits(3, static_cast<uint64_t>(n));
  bw->WriteBits(n, static_cast<uint64_t>(u));
}

// Reads U16 in [0,65535] using 1–21 bits (used for prefix alphabet sizes).
static int ReadVarLenUint16(BitReader& br) {
  if (!br.ReadBits(1)) return 0;
  int n = br.ReadBits(4);
  if (n == 0) return 1;
  return static_cast<int>(br.ReadBits(n)) + (1 << n);
}

// ── UintConfig ───────────────────────────────────────────────────────────────

static UintConfig ReadUintConfig(BitReader& br, int log_alpha_size) {
  UintConfig c;
  int bits = CeilLog2(log_alpha_size + 1);
  c.split_exponent = br.ReadBits(bits);
  if (static_cast<int>(c.split_exponent) != log_alpha_size) {
    int nb = CeilLog2(static_cast<int>(c.split_exponent) + 1);
    c.msb_in_token = br.ReadBits(nb);
    nb = CeilLog2(static_cast<int>(c.split_exponent) -
                  static_cast<int>(c.msb_in_token) + 1);
    c.lsb_in_token = br.ReadBits(nb);
  }
  c.split = 1u << c.split_exponent;
  return c;
}

// Expand a token to a hybrid-uint value.
static uint32_t ExpandToken(BitReader& br, const UintConfig& cfg,
                            uint32_t token) {
  if (token < cfg.split) return token;
  uint32_t n = cfg.split_exponent - cfg.msb_in_token - cfg.lsb_in_token +
               ((token - cfg.split) >> (cfg.msb_in_token + cfg.lsb_in_token));
  n &= 31u;
  uint32_t low = token & ((1u << cfg.lsb_in_token) - 1);
  token = (token >> cfg.lsb_in_token) & ((1u << cfg.msb_in_token) - 1);
  token |= (1u << cfg.msb_in_token);
  return (((token << n) | br.ReadBits(static_cast<int>(n)))
          << cfg.lsb_in_token) |
         low;
}

// ── Huffman table building
// ────────────────────────────────────────────────────

static bool BuildHuffTable(const std::vector<uint8_t>& lengths,
                           std::vector<HuffEntry>* primary,
                           std::vector<std::vector<HuffEntry>>* exts) {
  const int N = static_cast<int>(lengths.size());

  // Count per length.
  int cnt[kPrefixMaxBits + 1] = {};
  for (int l : lengths) {
    if (l > kPrefixMaxBits) return false;
    if (l > 0) ++cnt[l];
  }

  // Canonical starting codes (MSB-first).
  int start[kPrefixMaxBits + 2] = {};
  for (int l = 1; l <= kPrefixMaxBits; ++l)
    start[l + 1] = (start[l] + cnt[l]) << 1;

  primary->assign(1 << kPrefixHuffBits, HuffEntry{0, 0, 0});
  exts->clear();

  // First pass: find maximum extension table sizes per primary slot.
  std::vector<int> ext_max(1 << kPrefixHuffBits, 0);
  std::vector<int> ext_id(1 << kPrefixHuffBits, -1);
  int next_ext = 0;

  // Assign canonical codes and analyse their primary slots.
  std::vector<int> cur(start, start + kPrefixMaxBits + 1);
  for (int sym = 0; sym < N; ++sym) {
    int l = lengths[sym];
    if (!l) continue;
    uint32_t code = static_cast<uint32_t>(cur[l]++);
    // Reverse for LSB-first table.
    uint32_t rcode = RevBits(code, l);
    if (l <= kPrefixHuffBits) {
      // Fill primary table slots.
      int step = 1 << l;
      for (int j = static_cast<int>(rcode); j < (1 << kPrefixHuffBits); j += step) {
        (*primary)[j] = {static_cast<uint8_t>(l), 0,
                         static_cast<uint16_t>(sym)};
      }
    } else {
      int prim = static_cast<int>(rcode) & ((1 << kPrefixHuffBits) - 1);
      int ext_len = l - kPrefixHuffBits;
      if (ext_len > ext_max[prim]) ext_max[prim] = ext_len;
    }
  }

  // Allocate extension tables and mark primary entries.
  for (int prim = 0; prim < (1 << kPrefixHuffBits); ++prim) {
    if (ext_max[prim] > 0) {
      // Verify the primary slot wasn't already set by a short code.
      if ((*primary)[prim].nbits != 0) return false;
      ext_id[prim] = next_ext++;
      exts->push_back(
          std::vector<HuffEntry>(1 << ext_max[prim], HuffEntry{0, 0, 0}));
      (*primary)[prim] = {255, static_cast<uint8_t>(ext_max[prim]),
                          static_cast<uint16_t>(ext_id[prim])};
    }
  }

  // Second pass: fill extension tables.
  std::copy(start, start + kPrefixMaxBits + 1, cur.begin());
  for (int sym = 0; sym < N; ++sym) {
    int l = lengths[sym];
    if (!l || l <= kPrefixHuffBits) continue;
    uint32_t code = static_cast<uint32_t>(cur[l]++);
    uint32_t rcode = RevBits(code, l);
    int prim = static_cast<int>(rcode) & ((1 << kPrefixHuffBits) - 1);
    int ext_bits_val =
        static_cast<int>(rcode >> kPrefixHuffBits) & ((1 << ext_max[prim]) - 1);
    int ext_len = l - kPrefixHuffBits;
    int step = 1 << ext_len;
    auto& ext = (*exts)[ext_id[prim]];
    for (int j = ext_bits_val; j < (1 << ext_max[prim]); j += step) {
      ext[j] = {static_cast<uint8_t>(l), 0, static_cast<uint16_t>(sym)};
    }
  }
  return true;
}

// ── Prefix symbol decode
// ──────────────────────────────────────────────────────

static uint32_t DecodePrefixSym(BitReader& br, const DistFull& d) {
  if (d.degenerate_sym >= 0) return static_cast<uint32_t>(d.degenerate_sym);
  uint32_t bits7 = br.PeekBits(kPrefixHuffBits);
  const HuffEntry& e = d.huff[bits7];
  if (e.nbits == 0) {
    // Degenerate single-symbol entry: consume 0 bits.
    return e.value;
  }
  if (e.nbits == 255) {
    // Extension table.
    br.Consume(kPrefixHuffBits);
    uint32_t ext_bits = br.PeekBits(e.ext_bits);
    const HuffEntry& e2 = d.huff_ext[e.value][ext_bits];
    br.Consume(e2.nbits - kPrefixHuffBits);
    return e2.value;
  }
  br.Consume(e.nbits);
  return e.value;
}

// ── Brotli simple-code prefix reading ────────────────────────────────────────

static bool ReadSimplePrefix(BitReader& br, int alphabet_size, DistFull* d) {
  int max_bits = (alphabet_size > 1) ? CeilLog2(alphabet_size) : 0;
  int num_symbols = static_cast<int>(br.ReadBits(2)) + 1;
  uint16_t syms[5] = {};
  for (int i = 0; i < num_symbols; ++i) {
    syms[i] = static_cast<uint16_t>(br.ReadBits(max_bits));
    if (syms[i] >= alphabet_size) return false;
  }
  for (int i = 0; i < num_symbols - 1; ++i)
    for (int j = i + 1; j < num_symbols; ++j)
      if (syms[i] == syms[j]) return false;

  if (num_symbols == 4) num_symbols += static_cast<int>(br.ReadBits(1));

  if (num_symbols == 1) {
    d->degenerate_sym = syms[0];
    d->huff.assign(1 << kPrefixHuffBits, HuffEntry{0, 0, syms[0]});
    return true;
  }

  std::vector<uint8_t> lengths(alphabet_size, 0);
  switch (num_symbols) {
    case 2:
      if (syms[0] > syms[1]) std::swap(syms[0], syms[1]);
      lengths[syms[0]] = 1;
      lengths[syms[1]] = 1;
      break;
    case 3:
      if (syms[1] > syms[2]) std::swap(syms[1], syms[2]);
      lengths[syms[0]] = 1;
      lengths[syms[1]] = 2;
      lengths[syms[2]] = 2;
      break;
    case 4:
      for (int i = 0; i < 3; ++i)
        for (int j = i + 1; j < 4; ++j)
          if (syms[i] > syms[j]) std::swap(syms[i], syms[j]);
      lengths[syms[0]] = 2;
      lengths[syms[1]] = 2;
      lengths[syms[2]] = 2;
      lengths[syms[3]] = 2;
      break;
    case 5:
      if (syms[2] > syms[3]) std::swap(syms[2], syms[3]);
      lengths[syms[0]] = 1;
      lengths[syms[1]] = 2;
      lengths[syms[2]] = 3;
      lengths[syms[3]] = 3;
      break;
    default:
      return false;
  }
  return BuildHuffTable(lengths, &d->huff, &d->huff_ext);
}

// ── Brotli complex-prefix reading
// ─────────────────────────────────────────────

// Static 4-bit peek table for reading 18 code-length code lengths.
// Entry: {nbits, value}.  Matches libjxl
// HuffmanDecodingData::ReadFromBitStream.
static const uint8_t kClclNbits[16] = {2, 2, 2, 3, 2, 2, 2, 4,
                                       2, 2, 2, 3, 2, 2, 2, 4};
static const uint8_t kClclVal[16] = {0, 4, 3, 2, 0, 4, 3, 1,
                                     0, 4, 3, 2, 0, 4, 3, 5};
static const int kCodeLengthCodes = 18;
static const uint8_t kCodeLengthOrder[18] = {1, 2, 3, 4,  0,  5,  17, 6,  16,
                                             7, 8, 9, 10, 11, 12, 13, 14, 15};

// Decode one symbol from a small Huffman table (used for the code-length
// alphabet whose codes are at most 4 bits → primary table covers them.
// table with no extension tables needed).
static int DecodeCLSymbol(BitReader& br, const DistFull& d) {
  return static_cast<int>(DecodePrefixSym(br, d));
}

static bool ReadComplexPrefix(BitReader& br, int alphabet_size, int skip,
                              DistFull* d) {
  uint8_t clcl[kCodeLengthCodes] = {};
  int space = 32;
  int num_codes = 0;
  for (int i = skip; i < kCodeLengthCodes && space > 0; ++i) {
    int idx = kCodeLengthOrder[i];
    uint32_t bits4 = br.PeekBits(4);
    br.Consume(kClclNbits[bits4]);
    clcl[idx] = kClclVal[bits4];
    if (clcl[idx]) {
      space -= 32 >> clcl[idx];
      ++num_codes;
    }
  }
  if (num_codes != 1 && space != 0) return false;

  // Build small Huffman decoder for the 18 code-length symbols.
  std::vector<uint8_t> cl_lengths(clcl, clcl + kCodeLengthCodes);
  DistFull cl_dist;
  if (!BuildHuffTable(cl_lengths, &cl_dist.huff, &cl_dist.huff_ext))
    return false;

  // Read code lengths for all alphabet symbols.
  std::vector<uint8_t> lengths(alphabet_size, 0);
  int sym = 0;
  uint8_t prev_len = 8;
  int repeat = 0;
  uint8_t repeat_len = 0;
  int rem_space = 32768;

  while (sym < alphabet_size && rem_space > 0) {
    int code_len = DecodeCLSymbol(br, cl_dist);
    if (code_len < 16) {
      repeat = 0;
      lengths[sym++] = static_cast<uint8_t>(code_len);
      if (code_len) {
        prev_len = static_cast<uint8_t>(code_len);
        rem_space -= 32768 >> code_len;
      }
    } else {
      int extra_bits = code_len - 14;
      uint8_t new_len = (code_len == 16) ? prev_len : 0;
      if (repeat_len != new_len) {
        repeat = 0;
        repeat_len = new_len;
      }
      int old_repeat = repeat;
      if (repeat > 0) {
        repeat -= 2;
        repeat <<= extra_bits;
      }
      repeat += static_cast<int>(br.ReadBits(extra_bits)) + 3;
      int delta = repeat - old_repeat;
      if (sym + delta > alphabet_size) return false;
      memset(&lengths[sym], repeat_len, static_cast<size_t>(delta));
      sym += delta;
      if (repeat_len) rem_space -= delta * (32768 >> repeat_len);
    }
  }
  if (rem_space != 0) return false;
  return BuildHuffTable(lengths, &d->huff, &d->huff_ext);
}

// Read one prefix-code distribution from the bitstream.
static bool ReadPrefixDist(BitReader& br, int alphabet_size, DistFull* d) {
  if (alphabet_size <= 1) {
    d->degenerate_sym = 0;
    d->huff.assign(1 << kPrefixHuffBits, HuffEntry{0, 0, 0});
    return true;
  }
  int kind = static_cast<int>(br.ReadBits(2));
  if (kind == 1) return ReadSimplePrefix(br, alphabet_size, d);
  return ReadComplexPrefix(br, alphabet_size, kind, d);
}

// ── ANS distribution reading ─────────────────────────────────────────────────
// `precision_bits` = kANSLogTabSize = 12 for JXL.

// Hardcoded 128-entry (7-bit peek) table for ANS logcount symbols.
// [nbits_consumed, logcount_plus1].  logcount+1=0 means count=0 (skip).
// Matches libjxl dec_ans.cc huff[128][2].
static const uint8_t kLogcountHuff[128][2] = {
    {3, 10}, {7, 12}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
    {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
    {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
    {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
    {3, 10}, {6, 11}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
    {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
    {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
    {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
    {3, 10}, {7, 13}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
    {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
    {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
    {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
    {3, 10}, {6, 11}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
    {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
    {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
    {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
};

// PopCount precision (matches lib/jxl/ans_common.h GetPopulationCountPrecision).
static int PopPrecision(uint32_t logcount, uint32_t shift) {
  int32_t r = std::min<int32_t>(
      static_cast<int32_t>(logcount),
      static_cast<int32_t>(shift) -
          static_cast<int32_t>((kANSLogTabSize - logcount) >> 1));
  if (r < 0) return 0;
  return static_cast<int>(r);
}

static bool ReadANSDist(BitReader& br, std::vector<int32_t>* counts) {
  const size_t bits_in = br.pos();
  const int range = kANSTabSize;
  if (br.ReadBits(1)) {
    // Simple code.
    int num_sym = static_cast<int>(br.ReadBits(1)) + 1;
    int syms[2] = {};
    int max_sym = 0;
    for (int i = 0; i < num_sym; ++i) {
      syms[i] = ReadU8(br);
      if (syms[i] > max_sym) max_sym = syms[i];
    }
    counts->assign(max_sym + 1, 0);
    if (num_sym == 1) {
      (*counts)[syms[0]] = range;
    } else {
      if (syms[0] == syms[1]) return false;
      (*counts)[syms[0]] = static_cast<int32_t>(br.ReadBits(kANSLogTabSize));
      (*counts)[syms[1]] = range - (*counts)[syms[0]];
    }
    EntropyTrace(
        "ReadANSDist simple bits_in=%" PRIuS " bits_out=%" PRIuS
        " num_sym=%d sym0=%d sym1=%d c0=%" PRId32 " c1=%" PRId32,
        bits_in, br.pos(), num_sym, num_sym >= 1 ? syms[0] : -1,
        num_sym >= 2 ? syms[1] : -1,
        num_sym >= 1 && static_cast<size_t>(syms[0]) < counts->size()
            ? (*counts)[syms[0]]
            : -1,
        num_sym >= 2 && static_cast<size_t>(syms[1]) < counts->size()
            ? (*counts)[syms[1]]
            : -1);
    return true;
  }
  if (br.ReadBits(1)) {
    // Flat distribution.
    int alpha = ReadU8(br) + 1;
    if (alpha > range) return false;
    counts->assign(alpha, 0);
    int base = range / alpha, rem = range % alpha;
    for (int i = 0; i < alpha; ++i) (*counts)[i] = base + (i < rem ? 1 : 0);
    EntropyTrace("ReadANSDist flat bits_in=%" PRIuS " bits_out=%" PRIuS
                 " alpha=%d",
                 bits_in, br.pos(), alpha);
    return true;
  }

  // General case: read with the hardcoded logcount prefix code (same unary
  // bounding as lib/jxl/dec_ans.cc ReadHistogram).
  int len_log = 0;
  const int upper_bound_log = FloorLog2Nonzero(kANSLogTabSize + 1);
  for (; len_log < upper_bound_log; ++len_log)
    if (!br.ReadBits(1)) break;
  int shift = static_cast<int>((br.ReadBits(len_log) | (1u << len_log)) - 1);
  if (shift > kANSLogTabSize + 1) return false;

  int length = ReadU8(br) + 3;
  counts->assign(length, 0);
  std::vector<int> logcounts(length);  // like libjxl ReadHistogram: default 0
  std::vector<int> same(length, 0);
  int omit_log = -1, omit_pos = -1;

  for (int i = 0; i < length; ++i) {
    uint32_t idx = br.PeekBits(7);
    int nb = kLogcountHuff[idx][0];
    int lc = static_cast<int>(kLogcountHuff[idx][1]) - 1;
    br.Consume(nb);
    logcounts[i] = lc;  // -1 = count is 0; 0..12 = log2(count)
    if (lc == kANSLogTabSize) {
      int rle = ReadU8(br);
      same[i] = rle + 5;
      i += rle + 3;
      continue;
    }
    if (lc > omit_log) {
      omit_log = lc;
      omit_pos = i;
    }
  }
  if (omit_pos < 0) return false;
  if (omit_pos + 1 < length && logcounts[omit_pos + 1] == kANSLogTabSize)
    return false;

  int32_t total = 0;
  int numsame = 0;
  int32_t prev = 0;
  for (int i = 0; i < length; ++i) {
    if (same[i]) {
      numsame = same[i] - 1;
      prev = (i > 0) ? (*counts)[i - 1] : 0;
    }
    if (numsame > 0) {
      (*counts)[i] = prev;
      --numsame;
    } else {
      int code = logcounts[i];
      if (i == omit_pos || code < 0) continue;
      if (shift == 0 || code == 0) {
        (*counts)[i] = 1 << code;
      } else {
        int bc = PopPrecision(static_cast<uint32_t>(code),
                              static_cast<uint32_t>(shift));
        (*counts)[i] = (1 << code) +
                       (static_cast<int32_t>(br.ReadBits(bc)) << (code - bc));
      }
    }
    total += (*counts)[i];
  }
  (*counts)[omit_pos] = range - total;
  if ((*counts)[omit_pos] <= 0) return false;
  EntropyTrace(
      "ReadANSDist general bits_in=%" PRIuS " bits_out=%" PRIuS
      " shift=%d length=%d omit_pos=%d omit_count=%" PRId32,
      bits_in, br.pos(), shift, length, omit_pos, (*counts)[omit_pos]);
  return true;
}

// ── ANS alias table building ─────────────────────────────────────────────────

static bool BuildAliasTable(const std::vector<int32_t>& counts, int log_alpha,
                            std::vector<AliasEntry>* alias) {
  const int table_size = 1 << log_alpha;
  const int log_bucket = kANSLogTabSize - log_alpha;
  const int bucket_size = 1 << log_bucket;

  // Extend counts to table_size.
  std::vector<int32_t> c(table_size, 0);
  int64_t sum_counts = 0;
  for (int i = 0; i < (int)counts.size() && i < table_size; ++i) {
    c[i] = counts[i];
    sum_counts += counts[i];
  }
  if (sum_counts != kANSTabSize) return false;

  // Check for degenerate (single non-zero entry) — handled by caller.
  int nonzero = 0, nz_sym = 0;
  for (int i = 0; i < table_size; ++i)
    if (c[i]) {
      ++nonzero;
      nz_sym = i;
    }
  if (nonzero == 0) {
    c[0] = kANSTabSize;
    nonzero = 1;
    nz_sym = 0;
  }

  alias->resize(table_size);

  if (nonzero == 1) {
    for (int i = 0; i < table_size; ++i) {
      const uint32_t f0 = 0;
      const uint32_t f1 = static_cast<uint32_t>(c[nz_sym]);
      (*alias)[i] = {static_cast<uint16_t>(nz_sym),
                     0,
                     static_cast<uint32_t>(bucket_size * i),
                     static_cast<uint16_t>(f0),
                     static_cast<uint16_t>(f1 ^ f0)};
    }
    return true;
  }

  // Standard alias method.
  std::vector<int> cut(table_size), sym(table_size), off(table_size);
  std::vector<int> overfull, underfull;
  for (int i = 0; i < table_size; ++i) {
    cut[i] = c[i];
    sym[i] = i;
    off[i] = 0;
    if (c[i] > bucket_size)
      overfull.push_back(i);
    else if (c[i] < bucket_size)
      underfull.push_back(i);
  }
  while (!overfull.empty()) {
    if (underfull.empty()) return false;
    int o = overfull.back();
    overfull.pop_back();
    int u = underfull.back();
    underfull.pop_back();
    int by = bucket_size - cut[u];
    cut[o] -= by;
    sym[u] = o;
    off[u] = cut[o];
    if (cut[o] < bucket_size)
      underfull.push_back(o);
    else if (cut[o] > bucket_size)
      overfull.push_back(o);
  }
  for (int i = 0; i < table_size; ++i) {
    if (cut[i] == bucket_size) {
      sym[i] = i;
      off[i] = 0;
      cut[i] = 0;
    } else
      off[i] -= cut[i];
    const uint32_t f0 =
        (i < static_cast<int>(c.size())) ? static_cast<uint32_t>(c[i]) : 0u;
    const int i1 = sym[i];
    const uint32_t f1 = (i1 >= 0 && i1 < static_cast<int>(c.size()))
                            ? static_cast<uint32_t>(c[i1])
                            : 0u;
    (*alias)[i] = {static_cast<uint16_t>(sym[i]), static_cast<uint16_t>(cut[i]),
                   static_cast<uint32_t>(off[i]), static_cast<uint16_t>(f0),
                   static_cast<uint16_t>(f1 ^ f0)};
  }
  return true;
}

// ── ANS token decode
// ──────────────────────────────────────────────────────────

static uint32_t DecodeANSSym(BitReader& br, uint32_t* state,
                             const DistFull& d) {
  if (d.degenerate_sym >= 0) return static_cast<uint32_t>(d.degenerate_sym);
  int log_bucket = kANSLogTabSize - d.log_alpha;
  uint32_t res = *state & (kANSTabSize - 1);
  int i = static_cast<int>(res >> log_bucket);
  uint32_t pos = res & ((1u << log_bucket) - 1);
  const AliasEntry& e = d.alias[i];
  uint32_t sym = (pos >= e.cutoff) ? static_cast<uint32_t>(e.symbol)
                                   : static_cast<uint32_t>(i);
  uint32_t off = (pos >= e.cutoff) ? e.offset + pos : pos;
  const uint32_t freq1_xor_use =
      (pos >= e.cutoff) ? static_cast<uint32_t>(e.freq1_xor_freq0) : 0u;
  const uint32_t freq =
      static_cast<uint32_t>(e.freq0) ^ freq1_xor_use;
  *state = freq * (*state >> kANSLogTabSize) + off;
  // Match libjxl ANSSymbolReader::ReadSymbolANSWithoutRefill (branchless
  // normalize): peek 16 bits every step; consume them iff state < 2^16.
  const uint32_t new_state =
      (*state << 16u) | static_cast<uint32_t>(br.PeekBits(16));
  const bool normalize = *state < (1u << 16u);
  *state = normalize ? new_state : *state;
  if (normalize) {
    br.Consume(16);
  }
  return sym;
}

// ── Context-map decoding
// ──────────────────────────────────────────────────────

// Matches lib/jxl/dec_context_map.cc VerifyContextMap: every cluster index in
// [0, num_htrees) must appear at least once in the map.
static bool VerifyContextMap(const std::vector<uint8_t>& context_map,
                             size_t num_htrees) {
  if (num_htrees == 0) return false;
  std::vector<bool> have_htree(num_htrees, false);
  size_t num_found = 0;
  for (uint8_t htree : context_map) {
    if (htree >= num_htrees) return false;
    if (!have_htree[htree]) {
      have_htree[htree] = true;
      ++num_found;
    }
  }
  return num_found == num_htrees;
}

static bool DecodeContextMap(BitReader& br, size_t num_dist, bool disallow_lz77,
                             std::vector<uint8_t>* cmap, size_t* num_clusters) {
  const size_t bits0 = br.pos();
  cmap->resize(num_dist);
  if (num_dist == 1) {
    (*cmap)[0] = 0;
    *num_clusters = 1;
    EntropyTrace("DecodeContextMap trivial num_dist=1 bits=%" PRIuS, bits0);
    return true;
  }

  bool is_simple = br.ReadBits(1) != 0;
  if (is_simple) {
    int nbits = static_cast<int>(br.ReadBits(2));
    uint32_t max_cluster = 0;
    for (auto& v : *cmap) {
      v = static_cast<uint8_t>(br.ReadBits(nbits));
      if (v > max_cluster) max_cluster = v;
    }
    *num_clusters = max_cluster + 1;
    EntropyTrace(
        "DecodeContextMap simple bits0=%" PRIuS " bits1=%" PRIuS
        " num_dist=%" PRIuS " nbits=%d num_clusters=%" PRIuS,
        bits0, br.pos(), num_dist, nbits, *num_clusters);
    if (EntropyTraceEnabled() && num_dist <= 32) {
      for (size_t i = 0; i < num_dist; ++i) {
        EntropyTrace("  cmap[%" PRIuS "]=%u", i, (*cmap)[i]);
      }
    }
    if (!VerifyContextMap(*cmap, *num_clusters)) return false;
    return true;
  }

  bool use_mtf = br.ReadBits(1) != 0;
  EntropyTrace(
      "DecodeContextMap complex bits0=%" PRIuS " before_sub bits=%" PRIuS
      " num_dist=%" PRIuS " use_mtf=%d",
      bits0, br.pos(), num_dist, use_mtf ? 1 : 0);

  // Recursive: decode context map using 1-distribution entropy coder.
  CoderState sub;
  // If num_dist <= 2, disallow LZ77 in the sub-coder (per spec).
  if (!InitCoderState(br, 1, num_dist <= 2, &sub)) return false;

  uint32_t max_sym = 0;
  for (size_t i = 0; i < num_dist; ++i) {
    uint32_t sym = DecodeOne(br, sub, 0);
    if (sym > 255) return false;
    (*cmap)[i] = static_cast<uint8_t>(sym);
    if (sym > max_sym) max_sym = sym;
  }
  *num_clusters = max_sym + 1;

  if (!sub.use_prefix && sub.ans_state != kANSFinalState) return false;

  if (use_mtf) {
    // Inverse MTF: wire values are indices into the MTF array (same as
    // lib/jxl/dec_context_map.cc + inverse_mtf-inl.h). The update must use
    // that index, not the decoded cluster id.
    uint8_t mtf[256];
    for (int k = 0; k < 256; ++k) mtf[k] = static_cast<uint8_t>(k);
    for (auto& entry : *cmap) {
      const uint8_t index = entry;
      entry = mtf[index];
      if (index == 0) continue;
      for (int k = static_cast<int>(index); k > 0; --k) {
        mtf[k] = mtf[k - 1];
      }
      mtf[0] = entry;
    }
  }
  EntropyTrace(
      "DecodeContextMap done bits=%" PRIuS " num_clusters=%" PRIuS
      " use_prefix_sub=%d ans_state_sub=0x%08X",
      br.pos(), *num_clusters, sub.use_prefix ? 1 : 0, sub.ans_state);
  if (EntropyTraceEnabled() && num_dist <= 32) {
    for (size_t i = 0; i < num_dist; ++i) {
      EntropyTrace("  cmap[%" PRIuS "]=%u", i, (*cmap)[i]);
    }
  }
  if (!VerifyContextMap(*cmap, *num_clusters)) return false;
  return true;
}

// ── Full coder initialisation
// ─────────────────────────────────────────────────

static bool InitCoderState(BitReader& br, size_t num_dist, bool disallow_lz77,
                           CoderState* st) {
  const size_t bits_start = br.pos();
  st->num_to_copy = 0;
  st->copy_pos = 0;
  st->num_decoded = 0;

  EntropyTrace(
      "InitCoderState begin bits=%" PRIuS " num_dist=%" PRIuS
      " disallow_lz77=%d",
      bits_start, num_dist, disallow_lz77 ? 1 : 0);

  // Context map length (libjxl DecodeHistograms): when LZ77 is enabled, one
  // extra slot is appended; its value becomes the cluster index used for LZ77
  // distance symbols (context_map->back() in libjxl).
  size_t num_ctx = num_dist;

  // 1. LZ77 params.
  st->lz77.enabled = br.ReadBits(1) != 0;
  if (st->lz77.enabled) {
    if (disallow_lz77) return false;
    // LZ77Params::VisitFields — U32(Val(224), Val(512), Val(4096),
    // BitsOffset(15, 8), ...) for min_symbol.
    static const uint32_t kMinSymImm[] = {224, 512, 4096, 8};
    static const uint32_t kMinSymBits[] = {0, 0, 0, 15};
    uint32_t sel = br.ReadBits(2);
    st->lz77.min_symbol = kMinSymImm[sel] + br.ReadBits(kMinSymBits[sel]);
    static const uint32_t kMinLenImm[] = {3, 4, 5, 9};
    static const uint32_t kMinLenBits[] = {0, 0, 2, 8};
    sel = br.ReadBits(2);
    st->lz77.min_length = kMinLenImm[sel] + br.ReadBits(kMinLenBits[sel]);
    ++num_ctx;
    st->lz77.len_config = ReadUintConfig(br, 8);
    st->window = std::make_unique<uint32_t[]>(1u << 20);
    memset(st->window.get(), 0, sizeof(uint32_t) * (1u << 20));
    EntropyTrace(
        "InitCoderState lz77 bits=%" PRIuS " min_symbol=%u min_length=%u "
        "len_cfg split_exp=%u msb=%u lsb=%u",
        br.pos(), st->lz77.min_symbol, st->lz77.min_length,
        st->lz77.len_config.split_exponent, st->lz77.len_config.msb_in_token,
        st->lz77.len_config.lsb_in_token);
  }

  // 2. Clustering.
  size_t num_clusters = 0;
  if (!DecodeContextMap(br, num_ctx, disallow_lz77, &st->clusters,
                        &num_clusters))
    return false;
  if (st->lz77.enabled) {
    if (st->clusters.empty()) return false;
    // libjxl: code->lz77.nonserialized_distance_context = context_map->back();
    st->lz_dist_ctx = st->clusters.back();
  } else {
    st->lz_dist_ctx = 0;
  }
  EntropyTrace("InitCoderState after_context_map bits=%" PRIuS
               " num_clusters=%" PRIuS,
               br.pos(), num_clusters);

  // 3. use_prefix_code + log_alphabet_size.
  st->use_prefix = br.ReadBits(1) != 0;
  int log_alpha = st->use_prefix ? kPrefixMaxBits : (5 + (int)br.ReadBits(2));
  EntropyTrace("InitCoderState use_prefix=%d log_alpha=%d bits=%" PRIuS,
               st->use_prefix ? 1 : 0, log_alpha, br.pos());

  // 4. Per-cluster UintConfigs.
  st->dists.resize(num_clusters);
  for (size_t c = 0; c < num_clusters; ++c) {
    st->dists[c].use_prefix = st->use_prefix;
    st->dists[c].log_alpha = log_alpha;
    st->dists[c].config = ReadUintConfig(br, log_alpha);
    EntropyTrace(
        "InitCoderState uint_cfg cluster=%" PRIuS " bits=%" PRIuS
        " split_exp=%u msb=%u lsb=%u",
        c, br.pos(), st->dists[c].config.split_exponent,
        st->dists[c].config.msb_in_token, st->dists[c].config.lsb_in_token);
  }

  // 5. Per-cluster distributions.
  if (st->use_prefix) {
    std::vector<int> alpha_sizes(num_clusters);
    for (size_t c = 0; c < num_clusters; ++c)
      alpha_sizes[c] = ReadVarLenUint16(br) + 1;
    for (size_t c = 0; c < num_clusters; ++c) {
      if (!ReadPrefixDist(br, alpha_sizes[c], &st->dists[c])) return false;
    }
  } else {
    for (size_t c = 0; c < num_clusters; ++c) {
      const size_t bits_hist = br.pos();
      std::vector<int32_t> counts;
      if (!ReadANSDist(br, &counts)) {
        return false;
      }
      int nz = 0, nz_sym = 0;
      for (int s = 0; s < (int)counts.size(); ++s)
        if (counts[s]) {
          ++nz;
          nz_sym = s;
        }
      if (nz <= 1) {
        st->dists[c].degenerate_sym = (nz == 0) ? 0 : nz_sym;
      } else {
        st->dists[c].degenerate_sym = -1;
        if (!BuildAliasTable(counts, log_alpha, &st->dists[c].alias)) {
          return false;
        }
      }
      EntropyTrace(
          "InitCoderState ANS hist cluster=%" PRIuS " bits_hist=%" PRIuS
          " bits_after=%" PRIuS " alphabet=%" PRIuS " nz=%d degenerate=%d",
          c, bits_hist, br.pos(),
          static_cast<size_t>(counts.size()), nz,
          st->dists[c].degenerate_sym);
    }
    // Initial ANS state.
    st->ans_state = br.ReadBits(32);
    EntropyTrace("InitCoderState ANS initial_state=0x%08X bits=%" PRIuS,
                 st->ans_state, br.pos());
  }
  EntropyTrace("InitCoderState complete bits=%" PRIuS " delta_bits=%" PRIuS,
               br.pos(), br.pos() - bits_start);
  return true;
}

// ── Symbol decode
// ─────────────────────────────────────────────────────────────

static uint32_t DecodeOne(BitReader& br, CoderState& st, size_t ctx) {
  if (st.lz77.enabled && st.num_to_copy > 0) {
    uint32_t sym = st.window[st.copy_pos & 0xFFFFF];
    ++st.copy_pos;
    --st.num_to_copy;
    st.window[st.num_decoded & 0xFFFFF] = sym;
    ++st.num_decoded;
    return sym;
  }

  size_t cluster = (ctx < st.clusters.size()) ? st.clusters[ctx] : 0;
  DistFull& d = st.dists[cluster];

  uint32_t token = st.use_prefix ? DecodePrefixSym(br, d)
                                 : DecodeANSSym(br, &st.ans_state, d);

  // LZ77 (matches libjxl ANSSymbolReader::ReadHybridUintClusteredInlined):
  // compare raw token to threshold; length is expanded from (token - threshold)
  // using lz77_length_uint_config — no second ANS symbol for length.
  if (st.lz77.enabled && token >= st.lz77.min_symbol) {
    uint32_t llen =
        ExpandToken(br, st.lz77.len_config, token - st.lz77.min_symbol);
    st.num_to_copy = llen + st.lz77.min_length;

    size_t dcluster = st.lz_dist_ctx;
    if (dcluster >= st.dists.size()) dcluster = 0;
    DistFull& dd = st.dists[dcluster];
    uint32_t dt = st.use_prefix ? DecodePrefixSym(br, dd)
                                : DecodeANSSym(br, &st.ans_state, dd);
    uint32_t dist = ExpandToken(br, dd.config, dt);
    // dist_multiplier == 0: distance = dist + 1.
    dist += 1;
    dist = std::min(dist, st.num_decoded);
    dist = std::min(dist, 1u << 20);
    st.copy_pos = st.num_decoded - dist;

    uint32_t sym = st.window[st.copy_pos & 0xFFFFF];
    ++st.copy_pos;
    --st.num_to_copy;
    if (st.window) st.window[st.num_decoded & 0xFFFFF] = sym;
    ++st.num_decoded;
    return sym;
  }

  uint32_t sym = ExpandToken(br, d.config, token);

  if (st.window) st.window[st.num_decoded & 0xFFFFF] = sym;
  ++st.num_decoded;
  return sym;
}

// ── ICC context function (mirrors ICCANSContext from libjxl)
// ──────────────────

static uint8_t ByteKind1(uint8_t b) {
  if (b >= 'a' && b <= 'z') return 0;
  if (b >= 'A' && b <= 'Z') return 0;
  if (b >= '0' && b <= '9') return 1;
  if (b == '.' || b == ',') return 1;
  if (b == 0) return 2;
  if (b == 1) return 3;
  if (b < 16) return 4;
  if (b == 255) return 6;
  if (b > 240) return 5;
  return 7;
}
static uint8_t ByteKind2(uint8_t b) {
  if (b >= 'a' && b <= 'z') return 0;
  if (b >= 'A' && b <= 'Z') return 0;
  if (b >= '0' && b <= '9') return 1;
  if (b == '.' || b == ',') return 1;
  if (b < 16) return 2;
  if (b > 240) return 3;
  return 4;
}
static size_t ICCContext(size_t i, uint8_t b1, uint8_t b2) {
  if (i <= 128) return 0;
  return 1 + ByteKind1(b1) + ByteKind2(b2) * 8;
}

// ── TOC permutation entropy (minimal flat ANS + libjxl WriteTokens order)
// ──

static void WriteUintConfigHybrid000(BitWriter* bw, int log_alpha_size) {
  UintConfig c;
  c.split_exponent = 0;
  c.msb_in_token = 0;
  c.lsb_in_token = 0;
  c.split = 1;
  int bits = CeilLog2(log_alpha_size + 1);
  bw->WriteBits(bits, c.split_exponent);
  if (static_cast<int>(c.split_exponent) != log_alpha_size) {
    int nb = CeilLog2(static_cast<int>(c.split_exponent) + 1);
    bw->WriteBits(nb, c.msb_in_token);
    nb = CeilLog2(static_cast<int>(c.split_exponent) -
                  static_cast<int>(c.msb_in_token) + 1);
    bw->WriteBits(nb, c.lsb_in_token);
  }
}

static void WriteUintConfigBits(BitWriter* bw, const UintConfig& c,
                                int log_alpha_size) {
  int bits = CeilLog2(log_alpha_size + 1);
  bw->WriteBits(bits, c.split_exponent);
  if (static_cast<int>(c.split_exponent) != log_alpha_size) {
    int nb = CeilLog2(static_cast<int>(c.split_exponent) + 1);
    bw->WriteBits(nb, c.msb_in_token);
    nb = CeilLog2(static_cast<int>(c.split_exponent) -
                  static_cast<int>(c.msb_in_token) + 1);
    bw->WriteBits(nb, c.lsb_in_token);
  }
}

static UintConfig SplineHybridUintConfig() {
  UintConfig c{};
  // split=16, msb=lsb=0: token carries exponent class; mantissa in raw bits
  // (libjxl HybridUintConfig::Encode with split_exponent=4, msb=0, lsb=0).
  c.split_exponent = 4;
  c.msb_in_token = 0;
  c.lsb_in_token = 0;
  c.split = 1u << c.split_exponent;
  return c;
}

static uint32_t FloorLog2NonzeroU32(uint32_t v) {
  uint32_t b = 0;
  while ((v >>= 1u) != 0) {
    ++b;
  }
  return b;
}

// Same mapping as libjxl HybridUintConfig::Encode (dec_ans.h) — must agree with
// ExpandToken / libjxl ReadHybridUintConfig.
static bool HybridUintEncode(uint32_t value, const UintConfig& cfg,
                             uint32_t* out_tok, uint32_t* extra_bits, int* extra_n) {
  if (cfg.split_exponent < cfg.msb_in_token + cfg.lsb_in_token) {
    return false;
  }
  const uint32_t split_token = cfg.split;
  if (value < split_token) {
    *out_tok = value;
    *extra_bits = 0;
    *extra_n = 0;
    return true;
  }
  const uint32_t n = FloorLog2NonzeroU32(value);
  const uint32_t m = value - (1u << n);
  const uint32_t tok =
      split_token +
      ((n - cfg.split_exponent) << (cfg.msb_in_token + cfg.lsb_in_token)) +
      ((m >> (n - cfg.msb_in_token)) << cfg.lsb_in_token) +
      (m & ((1u << cfg.lsb_in_token) - 1u));
  const uint32_t nbits = n - cfg.msb_in_token - cfg.lsb_in_token;
  if (nbits > 31u) {
    return false;
  }
  const uint32_t bits =
      (nbits == 0)
          ? 0u
          : ((value >> cfg.lsb_in_token) &
             ((nbits >= 32u) ? ~0u : ((1u << nbits) - 1u)));
  if (tok >= 256u) {
    return false;
  }
  *out_tok = tok;
  *extra_bits = bits;
  *extra_n = static_cast<int>(nbits);
  return true;
}

struct SplineAnsTok {
  size_t ctx;
  uint32_t token;
  uint32_t nextra;
  uint32_t extrabits;
};

static bool PushSplineHybrid(std::vector<SplineAnsTok>* order, size_t ctx,
                             uint32_t sym_u, const UintConfig& cfg) {
  uint32_t tok;
  uint32_t eb;
  int en;
  if (!HybridUintEncode(sym_u, cfg, &tok, &eb, &en)) {
    return false;
  }
  if (tok >= 256) {
    return false;
  }
  order->push_back(SplineAnsTok{ctx, tok, static_cast<uint32_t>(en), eb});
  return true;
}

static bool PushSplinePack(std::vector<SplineAnsTok>* order, size_t ctx,
                         int32_t s, const UintConfig& cfg) {
  return PushSplineHybrid(order, ctx, PackSigned(s), cfg);
}

static void WriteFlatANSDist256(BitWriter* bw) {
  bw->WriteBits(1, 0);    // not "simple" ANS histogram
  bw->WriteBits(1, 1);   // flat
  WriteU8(bw, 255);       // alpha = ReadU8()+1 = 256
}

static void SymOffFromResidual(uint32_t res12, const DistFull& d,
                               uint32_t* sym_out, uint32_t* off_out) {
  int log_bucket = kANSLogTabSize - d.log_alpha;
  uint32_t bucket = res12 >> log_bucket;
  uint32_t pos = res12 & ((1u << log_bucket) - 1);
  const AliasEntry& e = d.alias[static_cast<int>(bucket)];
  *sym_out = (pos >= e.cutoff) ? e.symbol : bucket;
  *off_out = (pos >= e.cutoff) ? e.offset + pos : pos;
}

// Matches lib/jxl/enc_ans.cc ANSBuildInfoTable: for each residual i in
// [0, ANS_TAB_SIZE), Lookup yields (symbol, offset); reverse_map[symbol][offset]
// = i. Encode then uses state = (state/freq)<<12 + reverse[symbol][state%freq].
//
// Rows must be dense [0, freq(symbol)) like libjxl — do not sparse-resize by
// off alone or leading slots stay empty.
static bool BuildTocPermReverseMaps(const DistFull& d,
                                    std::vector<std::vector<uint16_t>>* rev) {
  // Sentinel must not equal any valid ANS residual index [0, kANSTabSize).
  constexpr uint16_t kUnset = 0xFFFFu;
  constexpr size_t kFlatSymFreq = 16;
  rev->assign(256, std::vector<uint16_t>(kFlatSymFreq, kUnset));
  for (uint32_t i = 0; i < kANSTabSize; ++i) {
    uint32_t sym, off;
    SymOffFromResidual(i, d, &sym, &off);
    if (sym >= 256 || off >= kFlatSymFreq) {
      return false;
    }
    uint16_t& slot = (*rev)[sym][off];
    if (slot != kUnset && slot != static_cast<uint16_t>(i)) {
      return false;
    }
    slot = static_cast<uint16_t>(i);
  }
  for (size_t sym = 0; sym < rev->size(); ++sym) {
    const auto& row = (*rev)[sym];
    for (size_t j = 0; j < row.size(); ++j) {
      if (row[j] == kUnset) {
        return false;
      }
    }
  }
  return true;
}

static uint32_t TocPutSymbol(uint32_t* st, uint16_t freq,
                             const std::vector<uint16_t>& rev, uint8_t* nbits) {
  uint32_t bits = 0;
  *nbits = 0;
  if ((*st >> (32 - kANSLogTabSize)) >= freq) {
    bits = *st & 0xffffu;
    *st >>= 16;
    *nbits = 16;
  }
  const uint32_t v = (*st / freq) << kANSLogTabSize;
  const uint32_t offset = rev[*st % freq];
  *st = v + offset;
  return bits;
}

// libjxl enc_ans.cc WriteTokens tail: uint64 addbits with 56-bit chunks, then
// Write(32, state) followed by Write(numallbits, allbits) and prior chunks in
// reverse push order. Per reverse-iteration addbits(extra) then addbits(ans)
// so LSB-first output reads ANS (and renorm) before hybrid extra bits.
static constexpr size_t kAnsTailMaxBitsPerAccum = 56;

struct AnsTailAccumulator {
  uint64_t allbits = 0;
  size_t numallbits = 0;
  std::vector<uint64_t> out;
  std::vector<size_t> out_nbits;

  void AddBits(uint64_t bits, size_t nbits) {
    if (nbits == 0) return;
    if (nbits < 64) {
      bits &= (1ull << nbits) - 1ull;
    }
    if (numallbits + nbits > kAnsTailMaxBitsPerAccum) {
      out.push_back(allbits);
      out_nbits.push_back(numallbits);
      numallbits = 0;
      allbits = 0;
    }
    allbits <<= nbits;
    allbits |= bits;
    numallbits += nbits;
  }

  void WriteAfterStateWord(BitWriter* bw, uint32_t state_word) const {
    bw->WriteBits(32, state_word);
    if (numallbits != 0) {
      bw->WriteBits(static_cast<int>(numallbits), allbits);
    }
    for (int i = static_cast<int>(out.size()) - 1; i >= 0; --i) {
      bw->WriteBits(static_cast<int>(out_nbits[static_cast<size_t>(i)]),
                    out[static_cast<size_t>(i)]);
    }
  }
};

}  // namespace

// ============================================================================
// Public API
// ============================================================================

EntropyCoder::~EntropyCoder() {
  delete static_cast<CoderState*>(internal);
  internal = nullptr;
}

bool InitEntropyCoder(BitReader& br, size_t num_dist, bool disallow_lz77,
                      EntropyCoder* ec) {
  auto* st = new CoderState();
  if (!InitCoderState(br, num_dist, disallow_lz77, st)) {
    delete st;
    return false;
  }
  ec->internal = st;
  return true;
}

uint32_t DecodeHybridUint(BitReader& br, EntropyCoder& ec, size_t ctx) {
  return DecodeOne(br, *static_cast<CoderState*>(ec.internal), ctx);
}

bool FinalizeEntropyCoder(EntropyCoder& ec) {
  auto* st = static_cast<CoderState*>(ec.internal);
  if (!st->use_prefix && st->ans_state != kANSFinalState) {
    EntropyTrace("FinalizeEntropyCoder bad ANS state=0x%08X want 0x%08X",
                 st->ans_state, kANSFinalState);
    delete st;
    ec.internal = nullptr;
    return false;
  }
  delete st;
  ec.internal = nullptr;
  return true;
}

bool SkipICCStream(BitReader& br, std::vector<uint8_t>* icc_bytes) {
  // Read encoded size using U64.
  uint64_t enc_size = ReadU64(br);
  if (enc_size > 268435456u) {
    fprintf(stderr,
            "jxltran: ICC profile encoded size too large (%" PRIu64 ")\n",
            enc_size);
    return false;
  }

  // Initialise entropy decoder.
  CoderState coder;
  if (!InitCoderState(br, kNumICCContexts, /*disallow_lz77=*/false, &coder)) {
    fprintf(stderr, "jxltran: failed to initialise ICC entropy coder\n");
    return false;
  }

  // Decode all enc_size bytes (keep them for ICC context computation).
  icc_bytes->resize(static_cast<size_t>(enc_size), 0);
  for (size_t i = 0; i < static_cast<size_t>(enc_size); ++i) {
    uint8_t b1 = (i > 0) ? (*icc_bytes)[i - 1] : 0;
    uint8_t b2 = (i > 1) ? (*icc_bytes)[i - 2] : 0;
    size_t ctx = ICCContext(i, b1, b2);
    uint32_t val = DecodeOne(br, coder, ctx);
    (*icc_bytes)[i] = static_cast<uint8_t>(val);
  }

  // Verify ANS final state.
  if (!coder.use_prefix && coder.ans_state != kANSFinalState) {
    fprintf(stderr, "jxltran: ICC entropy stream: bad ANS final state 0x%08X\n",
            coder.ans_state);
    return false;
  }

  return br.ok();
}

bool WriteTocPermutationAnsEntropy(BitWriter* bw,
                                   const std::vector<HybridTok>& toks) {
  DistFull d;
  d.use_prefix = false;
  d.log_alpha = 8;
  d.degenerate_sym = -1;
  d.config.split_exponent = 0;
  d.config.msb_in_token = 0;
  d.config.lsb_in_token = 0;
  d.config.split = 1;
  std::vector<int32_t> counts(256, 16);
  if (!BuildAliasTable(counts, 8, &d.alias)) {
    fprintf(stderr, "jxltran: internal error (TOC permutation ANS table)\n");
    return false;
  }
  std::vector<std::vector<uint16_t>> rev_by_sym;
  if (!BuildTocPermReverseMaps(d, &rev_by_sym)) {
    fprintf(stderr,
            "jxltran: internal error (TOC permutation ANS reverse map)\n");
    return false;
  }

  bw->WriteBits(1, 0);   // lz77 disabled
  bw->WriteBits(1, 1);   // simple context map
  bw->WriteBits(2, 0);   // nbits = 0 → eight implicit zeros
  bw->WriteBits(1, 0);   // use_prefix = false (ANS)
  bw->WriteBits(2, 3);   // log_alpha = 5 + 3 = 8
  WriteUintConfigHybrid000(bw, 8);
  WriteFlatANSDist256(bw);

  constexpr uint32_t kAnsSig = 0x13u;
  uint32_t state = kAnsSig << 16;

  AnsTailAccumulator tail;
  for (int i = static_cast<int>(toks.size()) - 1; i >= 0; --i) {
    const HybridTok& t = toks[static_cast<size_t>(i)];
    if (t.token >= 256) return false;
    if (t.nbits > 63) return false;
    uint8_t anb = 0;
    const std::vector<uint16_t>& rev = rev_by_sym[t.token];
    uint32_t ab = TocPutSymbol(&state, 16, rev, &anb);
    tail.AddBits(static_cast<uint64_t>(t.bits), t.nbits);
    tail.AddBits(static_cast<uint64_t>(ab), anb);
  }

  tail.WriteAfterStateWord(bw, state);
  return true;
}

bool EncodeSplinesBundleBits(const LfGlobalSplines& spl, size_t num_pixels,
                             std::vector<uint8_t>* out_bytes, size_t* out_bit_len,
                             bool pad_entropy_to_byte) {
  out_bytes->clear();
  if (out_bit_len != nullptr) {
    *out_bit_len = 0;
  }
  if (spl.splines.empty() ||
      spl.starting_points.size() != spl.splines.size()) {
    fprintf(stderr,
            "jxltran: EncodeSplinesBundleBits: empty splines or "
            "starting_points / splines size mismatch\n");
    return false;
  }

  constexpr size_t kCtxQA = 0;
  constexpr size_t kCtxStart = 1;
  constexpr size_t kCtxNumSplines = 2;
  constexpr size_t kCtxNumCP = 3;
  constexpr size_t kCtxCP = 4;
  constexpr size_t kCtxDCT = 5;

  constexpr size_t kMaxNumControlPoints = 1u << 20u;
  constexpr size_t kMaxNumControlPointsPerPixelRatio = 2;
  const size_t max_control_points =
      std::min(kMaxNumControlPoints,
               num_pixels / kMaxNumControlPointsPerPixelRatio);
  const size_t n_sp = spl.splines.size();
  if (n_sp > max_control_points || n_sp + 1 > max_control_points) {
    fprintf(stderr, "jxltran: EncodeSplinesBundleBits: too many splines\n");
    return false;
  }

  const UintConfig hcfg = SplineHybridUintConfig();
  std::vector<SplineAnsTok> order;
  order.reserve(4096);

  if (!PushSplineHybrid(&order, kCtxNumSplines,
                        static_cast<uint32_t>(n_sp - 1), hcfg)) {
    fprintf(stderr,
            "jxltran: EncodeSplinesBundleBits: num_splines hybrid failed\n");
    return false;
  }

  for (size_t i = 0; i < spl.starting_points.size(); ++i) {
    if (i == 0) {
      const int64_t x = static_cast<int64_t>(std::llround(
          static_cast<double>(spl.starting_points[i].first)));
      const int64_t y = static_cast<int64_t>(std::llround(
          static_cast<double>(spl.starting_points[i].second)));
      if (x < 0 || y < 0 || x > 0x7fffffffll || y > 0x7fffffffll) {
        fprintf(stderr,
                "jxltran: EncodeSplinesBundleBits: starting point out of "
                "uint32 range\n");
        return false;
      }
      if (!PushSplineHybrid(&order, kCtxStart, static_cast<uint32_t>(x),
                            hcfg) ||
          !PushSplineHybrid(&order, kCtxStart, static_cast<uint32_t>(y), hcfg)) {
        fprintf(stderr,
                "jxltran: EncodeSplinesBundleBits: starting point hybrid "
                "failed\n");
        return false;
      }
    } else {
      const int64_t px = static_cast<int64_t>(std::llround(
          static_cast<double>(spl.starting_points[i - 1].first)));
      const int64_t py = static_cast<int64_t>(std::llround(
          static_cast<double>(spl.starting_points[i - 1].second)));
      const int64_t cx = static_cast<int64_t>(std::llround(
          static_cast<double>(spl.starting_points[i].first)));
      const int64_t cy = static_cast<int64_t>(std::llround(
          static_cast<double>(spl.starting_points[i].second)));
      const int64_t dx = cx - px;
      const int64_t dy = cy - py;
      if (dx > INT32_MAX || dx < INT32_MIN || dy > INT32_MAX ||
          dy < INT32_MIN) {
        fprintf(stderr,
                "jxltran: EncodeSplinesBundleBits: starting delta overflow\n");
        return false;
      }
      if (!PushSplinePack(&order, kCtxStart, static_cast<int32_t>(dx), hcfg) ||
          !PushSplinePack(&order, kCtxStart, static_cast<int32_t>(dy), hcfg)) {
        fprintf(stderr,
                "jxltran: EncodeSplinesBundleBits: starting delta hybrid "
                "failed\n");
        return false;
      }
    }
  }

  if (!PushSplinePack(&order, kCtxQA, spl.quantization_adjustment, hcfg)) {
    fprintf(stderr,
            "jxltran: EncodeSplinesBundleBits: quantization_adjustment hybrid "
            "failed\n");
    return false;
  }

  size_t total_cp = n_sp;
  for (const QuantizedSplineData& sp : spl.splines) {
    if (sp.control_points.size() > max_control_points) {
      fprintf(stderr,
              "jxltran: EncodeSplinesBundleBits: too many control points\n");
      return false;
    }
    total_cp += sp.control_points.size();
    if (total_cp > max_control_points) {
      fprintf(stderr,
              "jxltran: EncodeSplinesBundleBits: total control points budget "
              "exceeded\n");
      return false;
    }
  }

  for (const QuantizedSplineData& sp : spl.splines) {
    if (!PushSplineHybrid(
            &order, kCtxNumCP,
            static_cast<uint32_t>(sp.control_points.size()), hcfg)) {
      fprintf(stderr,
              "jxltran: EncodeSplinesBundleBits: num_control_points hybrid "
              "failed\n");
      return false;
    }
    for (const std::pair<int64_t, int64_t>& d : sp.control_points) {
      if (d.first > INT32_MAX || d.first < INT32_MIN ||
          d.second > INT32_MAX || d.second < INT32_MIN) {
        fprintf(stderr,
                "jxltran: EncodeSplinesBundleBits: control point overflow\n");
        return false;
      }
      if (!PushSplinePack(&order, kCtxCP, static_cast<int32_t>(d.first),
                          hcfg) ||
          !PushSplinePack(&order, kCtxCP, static_cast<int32_t>(d.second),
                          hcfg)) {
        fprintf(stderr,
                "jxltran: EncodeSplinesBundleBits: control point hybrid "
                "failed\n");
        return false;
      }
    }
    for (size_t c = 0; c < 3; ++c) {
      for (size_t j = 0; j < 32; ++j) {
        if (!PushSplinePack(&order, kCtxDCT, sp.color_dct[c][j], hcfg)) {
          fprintf(stderr,
                  "jxltran: EncodeSplinesBundleBits: color DCT hybrid failed\n");
          return false;
        }
      }
    }
    for (size_t j = 0; j < 32; ++j) {
      if (!PushSplinePack(&order, kCtxDCT, sp.sigma_dct[j], hcfg)) {
        fprintf(stderr,
                "jxltran: EncodeSplinesBundleBits: sigma DCT hybrid failed\n");
        return false;
      }
    }
  }

  DistFull d;
  d.use_prefix = false;
  d.log_alpha = 8;
  d.degenerate_sym = -1;
  d.config = hcfg;
  std::vector<int32_t> counts(256, 16);
  if (!BuildAliasTable(counts, 8, &d.alias)) {
    fprintf(stderr, "jxltran: EncodeSplinesBundleBits: BuildAliasTable\n");
    return false;
  }
  std::vector<std::vector<uint16_t>> rev_by_sym;
  if (!BuildTocPermReverseMaps(d, &rev_by_sym)) {
    fprintf(stderr,
            "jxltran: EncodeSplinesBundleBits: BuildTocPermReverseMaps\n");
    return false;
  }

  BitWriter bw;
  bw.WriteBits(1, 0);    // lz77 disabled
  bw.WriteBits(1, 1);    // simple context map
  bw.WriteBits(2, 0);    // nbits = 0 → six ReadBits(0) cluster indices (no bits)
  bw.WriteBits(1, 0);    // use_prefix = false (ANS)
  bw.WriteBits(2, 3);    // log_alpha = 8
  WriteUintConfigBits(&bw, hcfg, 8);
  WriteFlatANSDist256(&bw);

  constexpr uint32_t kAnsSig = 0x13u;
  uint32_t state = kAnsSig << 16;

  AnsTailAccumulator tail;
  for (int k = static_cast<int>(order.size()) - 1; k >= 0; --k) {
    const SplineAnsTok& t = order[static_cast<size_t>(k)];
    (void)t.ctx;
    uint8_t anb = 0;
    if (t.token >= 256) {
      return false;
    }
    const std::vector<uint16_t>& rev = rev_by_sym[t.token];
    uint32_t ab = TocPutSymbol(&state, 16, rev, &anb);
    tail.AddBits(static_cast<uint64_t>(t.extrabits), t.nextra);
    tail.AddBits(static_cast<uint64_t>(ab), anb);
  }

  tail.WriteAfterStateWord(&bw, state);
  if (pad_entropy_to_byte) {
    bw.ZeroPadToByte();
  }
  const size_t nbits = bw.bit_pos();
  if (out_bit_len != nullptr) {
    *out_bit_len = nbits;
  }
  *out_bytes = bw.TakeBytes();
  if (out_bytes->size() * 8u < nbits) {
    return false;
  }
  if (out_bit_len != nullptr) {
    BitReader br_check(out_bytes->data(), out_bytes->size());
    LfGlobalSplines check{};
    if (!DecodeSplinesBundle(br_check, num_pixels, &check)) {
      fprintf(stderr,
              "jxltran: EncodeSplinesBundleBits: internal error: encoded bundle "
              "does not self-decode\n");
      return false;
    }
    *out_bit_len = br_check.pos();
  }
  return true;
}

}  // namespace jxltran
