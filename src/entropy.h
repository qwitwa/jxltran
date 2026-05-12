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

// ── Constants ──────────────────────────────────────────────────────────────

static constexpr int kANSLogTabSize = 12;
static constexpr uint32_t kANSTabSize = 1u << kANSLogTabSize;
static constexpr uint32_t kANSFinalState = 0x130000u;

// First-level Huffman lookup table width (7 bits → 128-entry table).
static constexpr int kHuffBits = 7;
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

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_ENTROPY_H_
