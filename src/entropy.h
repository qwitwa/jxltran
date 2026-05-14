// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Minimal entropy decoder for parsing JXL entropy-coded streams in jxltran.
// Supports ANS and prefix (Brotli-style Huffman) coding, LZ77, and hybrid
// unsigned integer decoding.  Only enough to advance a BitReader past an
// entropy-coded block — the decoded values are consumed but not stored (except
// for ICC, where we keep the last two bytes for context computation).

#ifndef TOOLS_JXLTRAN_ENTROPY_H_
#define TOOLS_JXLTRAN_ENTROPY_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "bits.h"

namespace jxltran {

struct LfGlobalSplines;

// ── Constants ──────────────────────────────────────────────────────────────

static constexpr int kANSLogTabSize = 12;
static constexpr uint32_t kANSTabSize = 1u << kANSLogTabSize;
static constexpr uint32_t kANSFinalState = 0x130000u;

// First-level Huffman lookup width for JXL prefix codes (matches
// lib/jxl/dec_huffman.h kHuffmanTableBits). ANS histogram log-count symbols use
// a separate hard-coded 7-bit peek in entropy.cc (ReadANSDist).
static constexpr int kPrefixHuffBits = 8;
static constexpr int kPrefixMaxBits = 15;

// Number of ICC prediction contexts (from icc_codec_common.h).
static constexpr size_t kNumICCContexts = 41;

// ── Hybrid-uint configuration ───────────────────────────────────────────────

struct UintConfig {
  uint32_t split_exponent = 0;
  uint32_t msb_in_token = 0;
  uint32_t lsb_in_token = 0;
  uint32_t split = 1;  // = 1 << split_exponent
};

// ── LZ77 parameters ─────────────────────────────────────────────────────────

struct Lz77Params {
  bool enabled = false;
  uint32_t min_symbol = 0;
  uint32_t min_length = 0;
  UintConfig len_config;
};

// ── Full entropy decoder state (opaque handle) ───────────────────────────────

struct EntropyCoder {
  void* internal = nullptr;
  ~EntropyCoder();
};

// ── Public API ───────────────────────────────────────────────────────────────

// Initialise an EntropyCoder from the bitstream.
// num_dist: number of pre-clustered probability distributions (contexts).
// disallow_lz77: true when decoding the context map for a small num_dist.
bool InitEntropyCoder(BitReader& br, size_t num_dist, bool disallow_lz77,
                      EntropyCoder* ec);

// Decode one hybrid-uint value from the entropy stream for the given context.
// The decoded value is returned; the bit reader is advanced accordingly.
uint32_t DecodeHybridUint(BitReader& br, EntropyCoder& ec, size_t ctx);

// Verify ANS end state (for ANS streams only).  Call after all symbols.
bool FinalizeEntropyCoder(EntropyCoder& ec);

// Skip (parse and discard) a complete ICC profile entropy-coded stream,
// including its leading U64 encoded-size field.  Returns false on parse error.
// On success *icc_bytes contains the decoded ICC bytes (needed for COPY).
bool SkipICCStream(BitReader& br, std::vector<uint8_t>* icc_bytes);

// Writes a TOC permutation entropy block compatible with InitEntropyCoder(...,
// 8, false) + DecodeHybridUint (split_exponent 0 hybrid-000). Uses a single
// flat ANS cluster (256 symbols × count 16, log_alpha = 8) — no histogram
// optimization — then ANS-encodes hybrid-uint tokens (|toks| in encode order:
// end token, then each Lehmer digit) using the same uint64 addbits + 56-bit
// chunk flush as libjxl WriteTokens (reverse over tokens; addbits calls hybrid
// extra then ANS renorm bits each step so LSB-first output reads ANS then extra).
// Initial ANS state word is written first, then the packed tail. Returns false on
// I/O or internal consistency errors.
struct HybridTok {
  uint32_t token;
  uint32_t nbits;
  uint32_t bits;
};

bool WriteTocPermutationAnsEntropy(BitWriter* bw,
                                   const std::vector<HybridTok>& toks);

// Re-encode LF-global splines: LZ77 off, simple context map mapping all six
// spline contexts to one cluster (so encode-time ANS matches decode-time alias
// tables; six clusters with identical counts can still get different alias layouts),
// flat 256-way ANS (log_alpha = 8), hybrid-uint split_exponent=4, msb=0, lsb=0
// (split=16; libjxl Encode-style mapping). ANS tail
// matches libjxl WriteTokens (reverse over tokens; addbits hybrid extra then ANS
// per step; 56-bit chunk flush; initial state word first on the wire).
// |out_bit_len| receives the exact number of bits written (trailing bits of the
// last stored byte are undefined unless |pad_entropy_to_byte| pads to a byte).
// When |pad_entropy_to_byte| is true, trailing zero bits are written until the
// bit length is a multiple of 8 (used when inserting a new bundle with an
// empty prior span so the body size delta stays whole-byte).
bool EncodeSplinesBundleBits(const LfGlobalSplines& spl, size_t num_pixels,
                             std::vector<uint8_t>* out_bytes, size_t* out_bit_len,
                             bool pad_entropy_to_byte = false);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_ENTROPY_H_
