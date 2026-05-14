// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Frame-level TOC permutation / group-order helpers (mirrors a subset of
// cjxl --group_order / --center_x / --center_y).

#ifndef TOOLS_JXLTRAN_TOC_GROUP_ORDER_H_
#define TOOLS_JXLTRAN_TOC_GROUP_ORDER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxltran {

struct ParsedCodestream;
struct FramedUnit;
struct FrameTocMetrics;

enum class TocGroupOrderCli {
  kKeep,
  kCanonical,    // 0: strip TOC permutation (logical stream order)
  kCenterFirst,  // 1: not implemented (requires permutation entropy encode)
  kProgressive,  // normalize only when HF data appears before all LF groups
};

bool TocStreamOrderLooksProgressive(const FramedUnit& fu,
                                    const FrameTocMetrics& tm);

// Applies |order| to eligible frames (optionally restricted by |only_frames|).
// kCanonical strips permutation; kProgressive strips only when not
// TocStreamOrderLooksProgressive. kCenterFirst is rejected (not implemented).
// When |any_mutation| is non-null, sets *any_mutation true if any frame was
// modified.
bool ApplyTocGroupOrder(ParsedCodestream* cs, TocGroupOrderCli order,
                        int64_t center_x, int64_t center_y,
                        const std::vector<size_t>* only_frames,
                        bool* any_mutation);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_TOC_GROUP_ORDER_H_
