// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "toc_group_order.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#include "codestream.h"
#include "frame_header.h"

namespace jxltran {

namespace {

static size_t AcGroupIndex(size_t pass, size_t group, size_t num_groups,
                           size_t num_dc_groups) {
  return 2 + num_dc_groups + pass * num_groups + group;
}

static bool InvertStreamPermutation(const std::vector<uint32_t>& perm,
                                    std::vector<size_t>* inv) {
  const size_t n = perm.size();
  inv->assign(n, n);
  for (size_t s = 0; s < n; ++s) {
    if (perm[s] >= n) return false;
    const size_t L = perm[s];
    if ((*inv)[L] != n) return false;
    (*inv)[L] = s;
  }
  for (size_t L = 0; L < n; ++L) {
    if ((*inv)[L] == n) return false;
  }
  return true;
}

static bool StripTocPermutationOnFrame(FramedUnit* fu) {
  if (fu->toc_perm.empty()) {
    return true;
  }
  const size_t n = fu->toc_perm.size();
  if (fu->toc_decoded_sizes.size() != n) {
    fprintf(stderr, "jxltran: TOC permutation/size count mismatch\n");
    return false;
  }
  fu->toc_strip_stream_sizes = fu->toc_decoded_sizes;
  std::vector<size_t> inv;
  if (!InvertStreamPermutation(fu->toc_perm, &inv)) {
    fprintf(stderr, "jxltran: invalid TOC permutation\n");
    return false;
  }
  std::vector<uint32_t> logical_sizes(n);
  for (size_t L = 0; L < n; ++L) {
    logical_sizes[L] = fu->toc_decoded_sizes[inv[L]];
  }
  fu->toc_decoded_sizes = std::move(logical_sizes);
  fu->toc_strip_logical_to_stream = std::move(inv);
  fu->toc_perm.clear();
  fu->toc_bits_before_sizes = 0;
  fu->toc_strip_perm_reorder = true;
  return true;
}

}  // namespace

bool TocStreamOrderLooksProgressive(const FramedUnit& fu,
                                    const FrameTocMetrics& tm) {
  if (fu.toc_decoded_sizes.empty()) return true;
  if (tm.num_toc_entries == 0) return true;
  if (fu.toc_decoded_sizes.size() != tm.num_toc_entries) return true;
  const size_t n = fu.toc_decoded_sizes.size();
  if (!fu.toc_perm.empty() && fu.toc_perm.size() != n) return true;

  const size_t ndc = tm.num_dc_groups;
  if (ndc >= 63) return true;
  const size_t first_hf_spatial =
      AcGroupIndex(0, 0, tm.num_groups, ndc);
  const uint64_t need_mask = (1ull << (ndc + 1)) - 1ull;
  uint64_t seen = 0;
  for (size_t s = 0; s < n; ++s) {
    const size_t L =
        fu.toc_perm.empty()
            ? s
            : (s < fu.toc_perm.size() ? static_cast<size_t>(fu.toc_perm[s]) : s);
    if (L <= ndc) {
      seen |= 1ull << L;
    } else if (L >= first_hf_spatial) {
      if ((seen & need_mask) != need_mask) {
        return false;
      }
    }
  }
  return true;
}

bool ApplyTocGroupOrder(ParsedCodestream* cs, TocGroupOrderCli order,
                        int64_t /*center_x*/, int64_t /*center_y*/,
                        const std::vector<size_t>* only_frames,
                        bool* any_mutation) {
  if (any_mutation != nullptr) *any_mutation = false;
  if (order == TocGroupOrderCli::kKeep) return true;
  if (order == TocGroupOrderCli::kCenterFirst) {
    fprintf(stderr,
            "jxltran: --group_order=1 (center-first) is not implemented yet "
            "(requires reordering frame body bytes to match a new TOC "
            "permutation).\n");
    return false;
  }

  const uint32_t cw = cs->image.size.width;
  const uint32_t ch = cs->image.size.height;

  const auto want = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;

    FrameTocMetrics tm;
    ComputeFrameTocMetrics(fu.frame, cs->image.metadata, cw, ch, &tm);
    if (tm.all_in_one_section || tm.num_toc_entries <= 1) {
      continue;
    }

    if (order == TocGroupOrderCli::kCanonical) {
      if (!fu.toc_perm.empty() && any_mutation != nullptr) {
        *any_mutation = true;
      }
      if (!StripTocPermutationOnFrame(&fu)) return false;
      continue;
    }
    if (order == TocGroupOrderCli::kProgressive) {
      if (!TocStreamOrderLooksProgressive(fu, tm)) {
        if (!fu.toc_perm.empty() && any_mutation != nullptr) {
          *any_mutation = true;
        }
        if (!StripTocPermutationOnFrame(&fu)) return false;
      }
    }
  }
  return true;
}

}  // namespace jxltran
