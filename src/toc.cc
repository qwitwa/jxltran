// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "toc.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <vector>

#include "bits.h"
#include "entropy.h"
#include "entropy_trace.h"
#include "printf_macros.h"
#include "trace.h"

namespace jxltran {

namespace {

// Same U32 distributions as lib/jxl/toc.h kTocDist.
static const U32Dist kTocDist[4] = {
    U32Dist::Bits(10),
    U32Dist::BitsOffset(14, 1024),
    U32Dist::BitsOffset(22, 17408),
    U32Dist::BitsOffset(30, 4211712),
};

static uint32_t ReadTocU32(BitReader& br) {
  return ReadU32(br, kTocDist[0], kTocDist[1], kTocDist[2], kTocDist[3]);
}

static bool JumpToByteBoundary(BitReader& br) {
  const size_t r = br.pos() & 7u;
  if (r == 0) return true;
  const int need = static_cast<int>(8u - r);
  for (int i = 0; i < need; ++i) {
    if (br.ReadBits(1) != 0) return false;
  }
  return br.ok();
}

static uint32_t FloorLog2Nonzero(uint32_t v) {
  uint32_t n = 0;
  while (v > 1u) {
    v >>= 1;
    ++n;
  }
  return n;
}

static void HybridUint000Encode(uint32_t value, uint32_t* token,
                                uint32_t* nbits, uint32_t* bits) {
  const uint32_t split_exponent = 0;
  const uint32_t split_token = 1u << split_exponent;
  const uint32_t msb_in_token = 0;
  const uint32_t lsb_in_token = 0;
  if (value < split_token) {
    *token = value;
    *nbits = 0;
    *bits = 0;
    return;
  }
  const uint32_t n = FloorLog2Nonzero(value);
  const uint32_t m = value - (1u << n);
  *token = split_token +
           ((n - split_exponent) << (msb_in_token + lsb_in_token)) +
           ((m >> (n - msb_in_token)) << lsb_in_token) +
           (m & ((1u << lsb_in_token) - 1));
  *nbits = n - msb_in_token - lsb_in_token;
  *bits = (value >> lsb_in_token) & ((1UL << *nbits) - 1);
}

static uint32_t CoeffOrderContext(uint32_t val) {
  uint32_t token, nbits, bits;
  HybridUint000Encode(val, &token, &nbits, &bits);
  constexpr uint32_t kPermutationContexts = 8;
  return std::min(token, kPermutationContexts - 1);
}

template <typename T>
static constexpr T ValueOfLowest1Bit(T t) {
  return t & -t;
}

static size_t CeilLog2Nonzero(size_t n) {
  size_t r = 0;
  size_t v = n - 1;
  while (v != 0) {
    v >>= 1;
    ++r;
  }
  return r;
}

// Decodes Lehmer code into a permutation (same algorithm as
// lib/jxl/lehmer_code.h).
static bool DecodeLehmerCode(const uint32_t* code, uint32_t* temp, size_t n,
                             uint32_t* permutation) {
  if (n == 0) return false;
  const size_t log2n = CeilLog2Nonzero(n);
  const size_t padded_n = 1ull << log2n;

  for (size_t i = 0; i < padded_n; i++) {
    const int32_t i1 = static_cast<int32_t>(i + 1);
    temp[i] = static_cast<uint32_t>(ValueOfLowest1Bit(i1));
  }

  for (size_t i = 0; i < n; i++) {
    if (code[i] + i >= n) return false;
    uint32_t rank = code[i] + 1;

    size_t bit = padded_n;
    size_t next = 0;
    for (size_t b = 0; b <= log2n; b++) {
      const size_t cand = next + bit;
      if (cand < 1) return false;
      bit >>= 1;
      if (temp[cand - 1] < rank) {
        next = cand;
        rank -= temp[cand - 1];
      }
    }

    permutation[i] = static_cast<uint32_t>(next);

    next += 1;
    while (next <= padded_n) {
      temp[next - 1] -= 1;
      next += ValueOfLowest1Bit(next);
    }
  }
  return true;
}

// Skips the TOC permutation entropy stream without storing the permutation.
static bool SkipTocPermutation(BitReader& br, size_t size) {
  EntropyCoder ec;
  if (!InitEntropyCoder(br, /*num_contexts=*/8, /*disallow_lz77=*/false, &ec)) {
    return false;
  }
  const uint32_t end =
      DecodeHybridUint(br, ec, CoeffOrderContext(static_cast<uint32_t>(size)));
  if (end > size) {
    (void)FinalizeEntropyCoder(ec);
    return false;
  }
  uint32_t last = 0;
  for (size_t i = 0; i < end; ++i) {
    const uint32_t lehmer = DecodeHybridUint(br, ec, CoeffOrderContext(last));
    last = lehmer;
    if (lehmer >= size - i) {
      (void)FinalizeEntropyCoder(ec);
      return false;
    }
  }
  return FinalizeEntropyCoder(ec);
}

// Decodes permutation into |perm| (size |size|).
static bool DecodeTocPermutation(BitReader& br, size_t size,
                                 std::vector<uint32_t>* perm) {
  const size_t bits0 = br.pos();
  EntropyTrace("DecodeTocPermutation begin bits=%" PRIuS " size=%" PRIuS,
               bits0, size);
  EntropyCoder ec;
  if (!InitEntropyCoder(br, /*num_contexts=*/8, /*disallow_lz77=*/false, &ec)) {
    return false;
  }
  EntropyTrace("DecodeTocPermutation after_InitEntropyCoder bits=%" PRIuS,
               br.pos());
  const uint32_t end =
      DecodeHybridUint(br, ec, CoeffOrderContext(static_cast<uint32_t>(size)));
  EntropyTrace("DecodeTocPermutation end_count=%u bits=%" PRIuS, end, br.pos());
  if (end > size) {
    (void)FinalizeEntropyCoder(ec);
    return false;
  }
  std::vector<uint32_t> lehmer(size, 0);
  uint32_t last = 0;
  for (size_t i = 0; i < end; ++i) {
    lehmer[i] = DecodeHybridUint(br, ec, CoeffOrderContext(last));
    last = lehmer[i];
    if (EntropyTraceEnabled() && i < 64) {
      EntropyTrace("  lehmer[%" PRIuS "]=%u ctx_last=%u bits=%" PRIuS, i,
                   lehmer[i], last, br.pos());
    }
    if (lehmer[i] >= size - i) {
      (void)FinalizeEntropyCoder(ec);
      return false;
    }
  }
  if (!FinalizeEntropyCoder(ec)) return false;
  EntropyTrace("DecodeTocPermutation after_finalize bits=%" PRIuS, br.pos());

  perm->resize(size);
  const size_t log2n = CeilLog2Nonzero(size);
  const size_t padded_n = 1ull << log2n;
  std::vector<uint32_t> temp(padded_n, 0);
  return DecodeLehmerCode(lehmer.data(), temp.data(), size, perm->data());
}

static bool ComputeLehmerCode(const uint32_t* permutation, size_t n,
                                std::vector<uint32_t>* lehmer) {
  std::vector<uint32_t> temp(n + 1, 0);
  lehmer->resize(n);
  for (size_t idx = 0; idx < n; ++idx) {
    const uint32_t s = permutation[idx];
    uint32_t penalty = 0;
    uint32_t i = s + 1;
    while (i != 0) {
      penalty += temp[i];
      i &= i - 1;
    }
    if (s < penalty) return false;
    (*lehmer)[idx] = s - penalty;
    i = s + 1;
    while (i < n + 1) {
      temp[i] += 1;
      i += static_cast<uint32_t>(ValueOfLowest1Bit(i));
    }
  }
  return true;
}

static bool IsValidPermutation(const std::vector<uint32_t>& p, size_t n) {
  if (p.size() != n) return false;
  std::vector<bool> seen(n, false);
  for (uint32_t v : p) {
    if (static_cast<size_t>(v) >= n || seen[v]) return false;
    seen[v] = true;
  }
  return true;
}

static bool BuildTocPermHybridToks(size_t toc_size,
                                   const std::vector<uint32_t>& perm,
                                   std::vector<HybridTok>* out) {
  std::vector<uint32_t> lehmer;
  if (!ComputeLehmerCode(perm.data(), toc_size, &lehmer)) return false;
  size_t end = toc_size;
  while (end > 0 && lehmer[end - 1] == 0) --end;
  out->clear();
  HybridTok first{};
  HybridUint000Encode(static_cast<uint32_t>(end), &first.token, &first.nbits,
                      &first.bits);
  out->push_back(first);
  for (size_t i = 0; i < end; ++i) {
    HybridTok t{};
    HybridUint000Encode(lehmer[i], &t.token, &t.nbits, &t.bits);
    out->push_back(t);
  }
  return true;
}

}  // namespace

bool ReadFullToc(BitReader& br, size_t toc_entries, uint64_t* total,
                 std::vector<uint32_t>* sizes_opt,
                 std::vector<uint32_t>* perm_opt,
                 size_t* bits_before_sizes_out) {
  *total = 0;
  if (toc_entries == 0 || toc_entries > 65536) return false;

  const size_t read_start = br.pos();
  {
    const size_t p = br.pos();
    const bool have_perm = br.ReadBool();
    EntropyTrace("ReadFullToc permutation_flag=%d bits=%" PRIuS,
                 have_perm ? 1 : 0, br.pos());
    TraceReadField(p, "read.toc.permutation_flag", "%s",
                   have_perm ? "true" : "false");
    if (have_perm) {
      if (perm_opt != nullptr) {
        if (!DecodeTocPermutation(br, toc_entries, perm_opt)) return false;
      } else {
        if (!SkipTocPermutation(br, toc_entries)) return false;
      }
    } else if (perm_opt != nullptr) {
      perm_opt->clear();
    }
  }
  if (!JumpToByteBoundary(br)) return false;
  TraceReadField(br.pos(), "read.toc.after_perm_pad", "(byte-aligned)");
  if (bits_before_sizes_out != nullptr) {
    *bits_before_sizes_out = br.pos() - read_start;
  }

  if (sizes_opt != nullptr) {
    sizes_opt->clear();
    sizes_opt->resize(toc_entries);
  }
  for (size_t i = 0; i < toc_entries; ++i) {
    const size_t p = br.pos();
    const uint32_t s = ReadTocU32(br);
    *total += s;
    if (sizes_opt != nullptr) (*sizes_opt)[i] = s;
    TraceReadField(p, "read.toc.size[]", "i=%" PRIuS " bytes=%" PRIu32 "", i,
                   s);
  }
  if (!JumpToByteBoundary(br)) return false;
  TraceReadField(br.pos(), "read.toc.after_sizes_pad", "(byte-aligned)");
  return br.ok();
}

bool ReadToc(BitReader& br, size_t toc_entries, uint64_t* total) {
  return ReadFullToc(br, toc_entries, total, nullptr, nullptr, nullptr);
}

bool WriteDecodedToc(BitWriter& bw, size_t /*header_bits_mod8*/,
                     const std::vector<uint32_t>& perm_u32,
                     const std::vector<uint32_t>& sizes_u32) {
  if (perm_u32.empty()) {
    bw.WriteBool(false);
    bw.ZeroPadToByte();
    for (uint32_t s : sizes_u32) {
      WriteU32(bw, s, kTocDist[0], kTocDist[1], kTocDist[2], kTocDist[3]);
    }
    bw.ZeroPadToByte();
    return true;
  }
  if (perm_u32.size() != sizes_u32.size()) {
    fprintf(stderr,
            "jxltran: TOC permutation length (%" PRIuS ") does not match TOC "
            "entry count (%" PRIuS ")\n",
            perm_u32.size(), sizes_u32.size());
    return false;
  }
  if (!IsValidPermutation(perm_u32, sizes_u32.size())) {
    fprintf(stderr, "jxltran: invalid TOC permutation for re-encode\n");
    return false;
  }
  std::vector<HybridTok> toks;
  if (!BuildTocPermHybridToks(sizes_u32.size(), perm_u32, &toks)) {
    fprintf(stderr, "jxltran: could not build Lehmer code for TOC permutation\n");
    return false;
  }
  bw.WriteBool(true);
  if (!WriteTocPermutationAnsEntropy(&bw, toks)) return false;
  bw.ZeroPadToByte();
  for (uint32_t s : sizes_u32) {
    WriteU32(bw, s, kTocDist[0], kTocDist[1], kTocDist[2], kTocDist[3]);
  }
  bw.ZeroPadToByte();
  return true;
}

bool WriteTocSizeList(BitWriter& bw, const std::vector<uint32_t>& sizes_u32) {
  for (uint32_t s : sizes_u32) {
    WriteU32(bw, s, kTocDist[0], kTocDist[1], kTocDist[2], kTocDist[3]);
  }
  bw.ZeroPadToByte();
  return true;
}

}  // namespace jxltran
