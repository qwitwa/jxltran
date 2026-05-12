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
#include <cstring>
#include <memory>
#include <vector>

#include "bits.h"

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

// ANS alias-table entry.  freq = D[symbol] (the probability count).
struct AliasEntry {
  uint16_t symbol;
  uint16_t cutoff;
  uint32_t offset;
  uint32_t freq;
};

// One probability distribution (ANS or prefix-code).
struct DistFull {
  bool use_prefix = false;
  int log_alpha = 0;
  int degenerate_sym = -1;  // -1 → not degenerate

  // ANS alias table (size = 1 << log_alpha).
  std::vector<AliasEntry> alias;

  // Prefix lookup table (size = 1 << kHuffBits) + optional extension tables.
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

  primary->assign(1 << kHuffBits, HuffEntry{0, 0, 0});
  exts->clear();

  // First pass: find maximum extension table sizes per primary slot.
  std::vector<int> ext_max(1 << kHuffBits, 0);
  std::vector<int> ext_id(1 << kHuffBits, -1);
  int next_ext = 0;

  // Assign canonical codes and analyse their primary slots.
  std::vector<int> cur(start, start + kPrefixMaxBits + 1);
  for (int sym = 0; sym < N; ++sym) {
    int l = lengths[sym];
    if (!l) continue;
    uint32_t code = static_cast<uint32_t>(cur[l]++);
    // Reverse for LSB-first table.
    uint32_t rcode = RevBits(code, l);
    if (l <= kHuffBits) {
      // Fill primary table slots.
      int step = 1 << l;
      for (int j = static_cast<int>(rcode); j < (1 << kHuffBits); j += step) {
        (*primary)[j] = {static_cast<uint8_t>(l), 0,
                         static_cast<uint16_t>(sym)};
      }
    } else {
      int prim = static_cast<int>(rcode) & ((1 << kHuffBits) - 1);
      int ext_len = l - kHuffBits;
      if (ext_len > ext_max[prim]) ext_max[prim] = ext_len;
    }
  }

  // Allocate extension tables and mark primary entries.
  for (int prim = 0; prim < (1 << kHuffBits); ++prim) {
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
    if (!l || l <= kHuffBits) continue;
    uint32_t code = static_cast<uint32_t>(cur[l]++);
    uint32_t rcode = RevBits(code, l);
    int prim = static_cast<int>(rcode) & ((1 << kHuffBits) - 1);
    int ext_bits_val =
        static_cast<int>(rcode >> kHuffBits) & ((1 << ext_max[prim]) - 1);
    int ext_len = l - kHuffBits;
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
  uint32_t bits7 = br.PeekBits(kHuffBits);
  const HuffEntry& e = d.huff[bits7];
  if (e.nbits == 0) {
    // Degenerate single-symbol entry: consume 0 bits.
    return e.value;
  }
  if (e.nbits == 255) {
    // Extension table.
    br.Consume(kHuffBits);
    uint32_t ext_bits = br.PeekBits(e.ext_bits);
    const HuffEntry& e2 = d.huff_ext[e.value][ext_bits];
    br.Consume(e2.nbits - kHuffBits);
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
    d->huff.assign(1 << kHuffBits, HuffEntry{0, 0, syms[0]});
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
// alphabet whose codes are at most 4 bits → fits entirely in kHuffBits=7
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
    d->huff.assign(1 << kHuffBits, HuffEntry{0, 0, 0});
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

// PopCount precision (mirrors GetPopulationCountPrecision from libjxl).
static int PopPrecision(int logcount, int shift) {
  int r = std::min(logcount,
                   shift - (static_cast<int>(kANSLogTabSize) - logcount) / 2);
  return r < 0 ? 0 : r;
}

static bool ReadANSDist(BitReader& br, std::vector<int32_t>* counts) {
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
    return true;
  }
  if (br.ReadBits(1)) {
    // Flat distribution.
    int alpha = ReadU8(br) + 1;
    if (alpha > range) return false;
    counts->assign(alpha, 0);
    int base = range / alpha, rem = range % alpha;
    for (int i = 0; i < alpha; ++i) (*counts)[i] = base + (i < rem ? 1 : 0);
    return true;
  }

  // General case: read with the hardcoded logcount prefix code.
  int len_log = 0;
  for (; len_log < CeilLog2(kANSLogTabSize + 1); ++len_log)
    if (!br.ReadBits(1)) break;
  int shift = static_cast<int>((br.ReadBits(len_log) | (1u << len_log)) - 1);
  if (shift > kANSLogTabSize + 1) return false;

  int length = ReadU8(br) + 3;
  counts->assign(length, 0);
  std::vector<int> logcounts(length, -2);  // -2 = unset
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
      int code = logcounts[i];  // code = floor(log2(count)), -1 = count 0
      if (i == omit_pos || code < 0) continue;
      if (code == 0 || shift == 0) {
        (*counts)[i] = 1 << code;
      } else {
        int bc = PopPrecision(code, shift);
        (*counts)[i] = (1 << code) +
                       (static_cast<int32_t>(br.ReadBits(bc)) << (code - bc));
      }
    }
    total += (*counts)[i];
  }
  (*counts)[omit_pos] = range - total;
  if ((*counts)[omit_pos] <= 0) return false;
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
  for (int i = 0; i < (int)counts.size() && i < table_size; ++i)
    c[i] = counts[i];

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
      (*alias)[i] = {static_cast<uint16_t>(nz_sym), 0,
                     static_cast<uint32_t>(bucket_size * i),
                     static_cast<uint32_t>(c[nz_sym])};
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
    (*alias)[i] = {static_cast<uint16_t>(sym[i]), static_cast<uint16_t>(cut[i]),
                   static_cast<uint32_t>(off[i]),
                   static_cast<uint32_t>(c[sym[i]])};
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
  uint32_t sym = (pos >= e.cutoff) ? e.symbol : static_cast<uint32_t>(i);
  uint32_t off = (pos >= e.cutoff) ? e.offset + pos : pos;
  *state = e.freq * (*state >> kANSLogTabSize) + off;
  if (*state < (1u << 16)) *state = (*state << 16) | br.ReadBits(16);
  return sym;
}

// ── Context-map decoding
// ──────────────────────────────────────────────────────

static bool DecodeContextMap(BitReader& br, size_t num_dist, bool disallow_lz77,
                             std::vector<uint8_t>* cmap, size_t* num_clusters) {
  cmap->resize(num_dist);
  if (num_dist == 1) {
    (*cmap)[0] = 0;
    *num_clusters = 1;
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
    return true;
  }

  bool use_mtf = br.ReadBits(1) != 0;

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
    uint8_t mtf[256];
    for (int k = 0; k < 256; ++k) mtf[k] = static_cast<uint8_t>(k);
    for (auto& v : *cmap) {
      uint8_t val = mtf[v];
      v = val;
      for (int k = static_cast<int>(val); k > 0; --k) mtf[k] = mtf[k - 1];
      mtf[0] = val;
    }
  }
  return true;
}

// ── Full coder initialisation
// ─────────────────────────────────────────────────

static bool InitCoderState(BitReader& br, size_t num_dist, bool disallow_lz77,
                           CoderState* st) {
  st->num_to_copy = 0;
  st->copy_pos = 0;
  st->num_decoded = 0;

  // 1. LZ77 params.
  st->lz77.enabled = br.ReadBits(1) != 0;
  if (st->lz77.enabled) {
    if (disallow_lz77) return false;
    static const uint32_t kMinSymImm[] = {224, 512, 4096, 0};
    static const uint32_t kMinSymBits[] = {0, 0, 0, 15};
    uint32_t sel = br.ReadBits(2);
    st->lz77.min_symbol = kMinSymImm[sel] + br.ReadBits(kMinSymBits[sel]);
    static const uint32_t kMinLenImm[] = {3, 4, 5, 9};
    static const uint32_t kMinLenBits[] = {0, 0, 2, 8};
    sel = br.ReadBits(2);
    st->lz77.min_length = kMinLenImm[sel] + br.ReadBits(kMinLenBits[sel]);
    st->lz_dist_ctx = num_dist++;
    st->lz77.len_config = ReadUintConfig(br, 8);
    st->window = std::make_unique<uint32_t[]>(1u << 20);
    memset(st->window.get(), 0, sizeof(uint32_t) * (1u << 20));
  }

  // 2. Clustering.
  size_t num_clusters = 0;
  if (!DecodeContextMap(br, num_dist, disallow_lz77, &st->clusters,
                        &num_clusters))
    return false;

  // 3. use_prefix_code + log_alphabet_size.
  st->use_prefix = br.ReadBits(1) != 0;
  int log_alpha = st->use_prefix ? kPrefixMaxBits : (5 + (int)br.ReadBits(2));

  // 4. Per-cluster UintConfigs.
  st->dists.resize(num_clusters);
  for (size_t c = 0; c < num_clusters; ++c) {
    st->dists[c].use_prefix = st->use_prefix;
    st->dists[c].log_alpha = log_alpha;
    st->dists[c].config = ReadUintConfig(br, log_alpha);
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
      std::vector<int32_t> counts;
      if (!ReadANSDist(br, &counts)) return false;
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
        if (!BuildAliasTable(counts, log_alpha, &st->dists[c].alias))
          return false;
      }
    }
    // Initial ANS state.
    st->ans_state = br.ReadBits(32);
  }
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
  uint32_t sym = ExpandToken(br, d.config, token);

  if (st.lz77.enabled && sym >= st.lz77.min_symbol) {
    // LZ77 length.
    size_t lcluster =
        (st.lz_dist_ctx > 0 && 0 < st.clusters.size()) ? st.clusters[0] : 0;
    (void)lcluster;
    // Length token comes from the same cluster as the symbol.
    uint32_t lt = st.use_prefix ? DecodePrefixSym(br, d)
                                : DecodeANSSym(br, &st.ans_state, d);
    uint32_t llen = ExpandToken(br, st.lz77.len_config, lt);
    st.num_to_copy = llen + st.lz77.min_length;

    // Distance token from lz_dist_ctx cluster.
    size_t dcluster =
        (st.lz_dist_ctx < st.clusters.size()) ? st.clusters[st.lz_dist_ctx] : 0;
    DistFull& dd = st.dists[dcluster];
    uint32_t dt = st.use_prefix ? DecodePrefixSym(br, dd)
                                : DecodeANSSym(br, &st.ans_state, dd);
    uint32_t dist = ExpandToken(br, dd.config, dt);
    // dist_multiplier == 0: distance = dist + 1.
    dist += 1;
    dist = std::min(dist, st.num_decoded);
    dist = std::min(dist, 1u << 20);
    st.copy_pos = st.num_decoded - dist;

    sym = st.window[st.copy_pos & 0xFFFFF];
    ++st.copy_pos;
    --st.num_to_copy;
  }

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
  if (!st->use_prefix && st->ans_state != kANSFinalState) return false;
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

}  // namespace jxltran
