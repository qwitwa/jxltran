// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "toc_layout.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <string>

#include "codestream.h"
#include "frame_header.h"
#include "printf_macros.h"

namespace jxltran {

namespace {

static constexpr size_t kBlockDim = 8;

static size_t AcGroupIndex(size_t pass, size_t group, size_t num_groups,
                           size_t num_dc_groups) {
  return 2 + num_dc_groups + pass * num_groups + group;
}

static std::string LogicalSectionLabel(const FrameTocMetrics& m,
                                       const FrameHeader& fh, size_t logical) {
  if (m.num_toc_entries != 0 && logical >= m.num_toc_entries) {
    return "invalid_logical";
  }
  if (m.all_in_one_section) {
    return "combined";
  }
  const size_t ndc = m.num_dc_groups;
  if (logical == 0) {
    return "lf_global";
  }
  if (logical >= 1 && logical <= ndc) {
    const size_t dci = logical - 1;
    const size_t gx = dci % m.xsize_dc_groups;
    const size_t gy = dci / m.xsize_dc_groups;
    const size_t x0b = gx * m.group_dim;
    const size_t y0b = gy * m.group_dim;
    const size_t wb =
        std::min(m.group_dim, m.xsize_blocks > x0b ? m.xsize_blocks - x0b : 0u);
    const size_t hb =
        std::min(m.group_dim, m.ysize_blocks > y0b ? m.ysize_blocks - y0b : 0u);
    const size_t x0p = x0b * kBlockDim;
    const size_t y0p = y0b * kBlockDim;
    const size_t wp = wb * kBlockDim;
    const size_t hp = hb * kBlockDim;
    char buf[160];
    snprintf(buf, sizeof(buf),
             "lf_group %" PRIuS " (%" PRIuS "x%" PRIuS "+%" PRIuS "+%" PRIuS ")",
             dci, wp, hp, x0p, y0p);
    return std::string(buf);
  }
  const size_t global_ac = ndc + 1;
  if (logical == global_ac) {
    return fh.encoding == kFrameEncVarDCT ? "hf_global" : "pass_global";
  }
  const size_t ac0 = AcGroupIndex(0, 0, m.num_groups, ndc);
  const size_t rel = logical - ac0;
  const size_t ng = m.num_groups;
  const size_t pass = rel / ng;
  const size_t grp = rel % ng;
  const size_t gx = grp % m.xsize_groups;
  const size_t gy = grp / m.xsize_groups;
  const size_t x0 = gx * m.group_dim;
  const size_t y0 = gy * m.group_dim;
  const size_t w =
      std::min(m.group_dim, m.xsize > x0 ? m.xsize - x0 : 0u);
  const size_t h =
      std::min(m.group_dim, m.ysize > y0 ? m.ysize - y0 : 0u);
  char buf[200];
  snprintf(buf, sizeof(buf),
           "hf_pass %" PRIuS " group %" PRIuS " (%" PRIuS "x%" PRIuS "+%" PRIuS
           "+%" PRIuS ")",
           pass, grp, w, h, x0, y0);
  return std::string(buf);
}

}  // namespace

void PrintCodestreamTocLayoutVerbose(FILE* out, const ParsedCodestream& cs) {
  const uint32_t cw = cs.image.size.width;
  const uint32_t ch = cs.image.size.height;

  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    const FramedUnit& fu = cs.frames[fi];
    FrameTocMetrics tm;
    ComputeFrameTocMetrics(fu.frame, cs.image.metadata, cw, ch, &tm);

    fprintf(out, "toc_verbose: frame %" PRIuS " toc_entries=%" PRIuS
                " permutation=%s num_groups=%" PRIuS " num_dc_groups=%" PRIuS
                " num_passes=%" PRIuS " group_dim=%" PRIuS "\n",
            fi, tm.num_toc_entries,
            (!fu.toc_perm.empty() ? "yes" : "no"), tm.num_groups,
            tm.num_dc_groups, tm.num_passes, tm.group_dim);

    if (fu.toc_decoded_sizes.empty()) {
      fprintf(out, "toc_verbose: frame %" PRIuS ": (no decoded TOC sizes)\n",
              fi);
      continue;
    }
    if (fu.toc_decoded_sizes.size() != tm.num_toc_entries) {
      fprintf(out,
              "toc_verbose: frame %" PRIuS ": WARNING sizes.size=%" PRIuS
              " != expected toc_entries=%" PRIuS "\n",
              fi, fu.toc_decoded_sizes.size(), tm.num_toc_entries);
    }
    if (!fu.toc_perm.empty() && fu.toc_perm.size() != fu.toc_decoded_sizes.size()) {
      fprintf(out,
              "toc_verbose: frame %" PRIuS ": WARNING perm.size=%" PRIuS
              " != sizes.size=%" PRIuS "\n",
              fi, fu.toc_perm.size(), fu.toc_decoded_sizes.size());
    }

    const size_t n = std::min(fu.toc_decoded_sizes.size(), tm.num_toc_entries);
    const size_t body0_byte =
        fu.original_frame_byte_offset +
        (fu.toc_start_bit + fu.toc_bit_length) / 8u;
    size_t body_cursor = 0;
    for (size_t stream = 0; stream < n; ++stream) {
      const uint32_t sz = fu.toc_decoded_sizes[stream];
      const size_t logical =
          fu.toc_perm.empty() ? stream
                              : (stream < fu.toc_perm.size() ? fu.toc_perm[stream]
                                                              : stream);
      const size_t abs_byte = body0_byte + body_cursor;
      const std::string label = LogicalSectionLabel(tm, fu.frame, logical);
      fprintf(out,
              "toc_verbose: frame %" PRIuS " offset %" PRIuS
              " stream[%" PRIuS "] logical[%" PRIuS "] size %" PRIu32 " %s\n",
              fi, abs_byte, stream, logical, sz, label.c_str());
      body_cursor += sz;
    }
  }
}

}  // namespace jxltran
