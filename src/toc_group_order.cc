// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "toc_group_order.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <utility>
#include <vector>

#include "codestream.h"
#include "frame_header.h"
#include "printf_macros.h"

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
  fu->toc_body_stream_shuffle.clear();
  fu->toc_body_shuffle_src_sizes.clear();
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

static constexpr double kPi = 3.1415926535897932384626433832795;

// Permutation of stream indices 0..n-1: chunk at old linear index |src| moves
// to stream position perm_move[src] (same convention as libjxl PermuteGroups).
static bool BuildLibjxlMovePermutation(const FrameTocMetrics& tm, int64_t center_x,
                                     int64_t center_y,
                                     std::vector<size_t>* perm_move) {
  perm_move->clear();
  const size_t ndc = tm.num_dc_groups;
  const size_t ng = tm.num_groups;
  const size_t np = tm.num_passes;
  if (tm.all_in_one_section || tm.num_toc_entries <= 1) {
    return true;
  }
  if (ng == 1 && np == 1) {
    return true;
  }
  const size_t group_dim = tm.group_dim;
  if (group_dim == 0) {
    return false;
  }

  int64_t imag_cx =
      (center_x >= 0 && static_cast<size_t>(center_x) < tm.xsize)
          ? center_x
          : static_cast<int64_t>(tm.xsize / 2);
  int64_t imag_cy =
      (center_y >= 0 && static_cast<size_t>(center_y) < tm.ysize)
          ? center_y
          : static_cast<int64_t>(tm.ysize / 2);

  const int64_t cx =
      (imag_cx / static_cast<int64_t>(group_dim)) *
          static_cast<int64_t>(group_dim) +
      static_cast<int64_t>(group_dim / 2);
  const int64_t cy =
      (imag_cy / static_cast<int64_t>(group_dim)) *
          static_cast<int64_t>(group_dim) +
      static_cast<int64_t>(group_dim / 2);

  const double direction =
      -std::atan2(static_cast<double>(imag_cy - cy),
                  static_cast<double>(imag_cx - cx));
  const double side_d =
      std::fmod((direction + 5 * kPi / 4), 2 * kPi) * 2 / kPi;
  const int64_t side = static_cast<int64_t>(side_d);

  auto group_center = [&](size_t gid, int64_t* gcx, int64_t* gcy) {
    const size_t gx = gid % tm.xsize_groups;
    const size_t gy = gid / tm.xsize_groups;
    const int64_t x0 = static_cast<int64_t>(gx * group_dim);
    const int64_t y0 = static_cast<int64_t>(gy * group_dim);
    *gcx = x0 + static_cast<int64_t>(group_dim / 2);
    *gcy = y0 + static_cast<int64_t>(group_dim / 2);
  };

  auto distance_key = [&](size_t gid) {
    int64_t gcx, gcy;
    group_center(gid, &gcx, &gcy);
    const int64_t dx = gcx - cx;
    const int64_t dy = gcy - cy;
    const int64_t adx = dx >= 0 ? dx : -dx;
    const int64_t ady = dy >= 0 ? dy : -dy;
    const double angle =
        std::remainder(std::atan2(static_cast<double>(dy), static_cast<double>(dx)) +
                           kPi / 4 + static_cast<double>(side) * (kPi / 2),
                       2 * kPi);
    return std::make_pair(std::max(adx, ady), angle);
  };

  std::vector<size_t> ac_order(ng);
  std::iota(ac_order.begin(), ac_order.end(), 0);
  std::sort(ac_order.begin(), ac_order.end(),
            [&](size_t a, size_t b) { return distance_key(a) < distance_key(b); });

  std::vector<size_t> inv_ac(ng, 0);
  for (size_t i = 0; i < ac_order.size(); ++i) {
    inv_ac[ac_order[i]] = i;
  }

  perm_move->reserve(ndc + 2 + ng * np);
  for (size_t i = 0; i < ndc + 2; ++i) {
    perm_move->push_back(i);
  }
  for (size_t pass = 0; pass < np; ++pass) {
    const size_t pass_start = perm_move->size();
    for (size_t gid = 0; gid < ng; ++gid) {
      perm_move->push_back(pass_start + inv_ac[gid]);
    }
  }
  if (perm_move->size() != tm.num_toc_entries) {
    fprintf(stderr,
            "jxltran: internal error: center-first permutation size %" PRIuS
            " != toc_entries %" PRIuS "\n",
            perm_move->size(), tm.num_toc_entries);
    return false;
  }
  return true;
}

static bool ApplyCenterFirstOnFrame(FramedUnit* fu, const FrameTocMetrics& tm,
                                    int64_t center_x, int64_t center_y,
                                    bool* mutated) {
  fu->toc_body_stream_shuffle.clear();
  fu->toc_body_shuffle_src_sizes.clear();

  if (tm.all_in_one_section || tm.num_toc_entries <= 1) {
    return true;
  }
  const size_t n = fu->toc_decoded_sizes.size();
  if (n == 0 || n != tm.num_toc_entries) {
    fprintf(stderr,
            "jxltran: --group_order=1: TOC size mismatch (got %" PRIuS
            ", expected %" PRIuS ")\n",
            fu->toc_decoded_sizes.size(), tm.num_toc_entries);
    return false;
  }
  if (fu->toc_strip_perm_reorder) {
    fprintf(stderr,
            "jxltran: --group_order=1 cannot combine with another TOC strip on "
            "the same frame.\n");
    return false;
  }

  std::vector<size_t> perm_move;
  if (!BuildLibjxlMovePermutation(tm, center_x, center_y, &perm_move)) {
    return false;
  }
  if (perm_move.empty()) {
    return true;
  }

  std::vector<size_t> dest_to_src(n, n);
  for (size_t src = 0; src < n; ++src) {
    const size_t d = perm_move[src];
    if (d >= n || dest_to_src[d] != n) {
      fprintf(stderr, "jxltran: --group_order=1: invalid move permutation\n");
      return false;
    }
    dest_to_src[d] = src;
  }

  bool identity_shuffle = true;
  for (size_t i = 0; i < n; ++i) {
    if (dest_to_src[i] != i) {
      identity_shuffle = false;
      break;
    }
  }
  if (identity_shuffle) {
    return true;
  }

  const std::vector<uint32_t> src_sizes = fu->toc_decoded_sizes;
  const std::vector<uint32_t> old_perm = fu->toc_perm;
  auto logical_at_stream = [&](size_t stream_slot) -> size_t {
    if (old_perm.empty()) {
      return stream_slot;
    }
    if (stream_slot >= old_perm.size()) {
      return stream_slot;
    }
    return static_cast<size_t>(old_perm[stream_slot]);
  };

  std::vector<uint32_t> new_perm(n, 0);
  std::vector<uint32_t> new_sizes(n, 0);
  for (size_t new_s = 0; new_s < n; ++new_s) {
    const size_t old_s = dest_to_src[new_s];
    new_sizes[new_s] = src_sizes[old_s];
    new_perm[new_s] = static_cast<uint32_t>(logical_at_stream(old_s));
  }

  fu->toc_decoded_sizes = std::move(new_sizes);
  fu->toc_perm = std::move(new_perm);
  fu->toc_bits_before_sizes = 0;
  fu->toc_strip_perm_reorder = false;
  fu->toc_strip_stream_sizes.clear();
  fu->toc_strip_logical_to_stream.clear();

  fu->toc_body_shuffle_src_sizes = src_sizes;
  fu->toc_body_stream_shuffle = std::move(dest_to_src);
  if (mutated != nullptr) {
    *mutated = true;
  }
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
                        int64_t center_x, int64_t center_y,
                        const std::vector<size_t>* only_frames,
                        bool* any_mutation) {
  if (any_mutation != nullptr) *any_mutation = false;
  if (order == TocGroupOrderCli::kKeep) return true;

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

    if (order == TocGroupOrderCli::kCenterFirst) {
      bool m = false;
      if (!ApplyCenterFirstOnFrame(&fu, tm, center_x, center_y, &m)) {
        return false;
      }
      if (m && any_mutation != nullptr) {
        *any_mutation = true;
      }
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
