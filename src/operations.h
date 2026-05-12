// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Higher-level codestream operations: mutate a ParsedCodestream in memory.

#ifndef TOOLS_JXLTRAN_OPERATIONS_H_
#define TOOLS_JXLTRAN_OPERATIONS_H_

#include <cstdint>

#include "codestream.h"

namespace jxltran {

struct GabArgs {
  enum class Kind { kNone, kBlur, kSharpen, kCustom };
  Kind kind = Kind::kNone;
  // kBlur / kSharpen: scales implicit encoder defaults (>= 0).
  float amount = 0.f;
  // kCustom: gab_x_weight1, gab_x_weight2, gab_y_*, gab_b_* (unnormalized).
  float custom[6] = {};
};

struct HeaderMod {
  uint32_t set_orientation = 0;
  uint32_t set_bits_per_sample = 0;
  bool have_set_num_loops = false;
  uint32_t set_num_loops = 0;
  bool have_set_tps = false;
  uint32_t set_tps_numerator = 1;
  uint32_t set_tps_denominator = 1;
};

// Applies header-only modifications to |cs| (image header fields).
bool ApplyHeaderMod(ParsedCodestream* cs, const HeaderMod& mod);

// Reversible metadata-only crop / canvas resize. (dx,dy) are relative to the
// current canvas: output pixel (px,py) shows input (px+dx, py+dy). New canvas
// size is (new_w x new_h). Encoded frame dimensions are unchanged; only
// canvas size and Regular / SkipProgressive frame origins (and have_crop when
// needed) are updated. LF and ReferenceOnly frame headers are not modified.
bool ApplyCrop(ParsedCodestream* cs, int32_t dx, int32_t dy, uint32_t new_w,
               uint32_t new_h);

// Reversible Gaborish (restoration filter) weight edits on the codestream
// headers. LF and reference-only frames are skipped. Decoded pixels change
// when the effective kernel changes.
bool ApplyGabArgs(ParsedCodestream* cs, const GabArgs& args);

// Photon noise: |change| false = leave bitstream noise as-is. |iso| == 0
// clears kNoise and removes the 80-bit LUT from DC global (when present).
// |iso| > 0 sets kNoise and LUT from the same ISO model as libjxl
// (SimulatePhotonNoise). Requires frames without patches or splines; TOC
// permutation is rejected when the body size changes.
bool ApplyPhotonNoiseIso(ParsedCodestream* cs, bool change, float iso);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_OPERATIONS_H_
