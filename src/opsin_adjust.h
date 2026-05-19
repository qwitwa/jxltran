// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Reversible XYB→linear-RGB tweaks via a non-default OpsinInverseMatrix
// (CustomTransformData), using the same defaults as libjxl.

#ifndef TOOLS_JXLTRAN_OPSIN_ADJUST_H_
#define TOOLS_JXLTRAN_OPSIN_ADJUST_H_

#include "codestream.h"

namespace jxltran {

// Optional float delta (used for opsin_biases[3] and quant_biases[4] tweaks).
struct OpsinBiasChannelOpt {
  bool set = false;
  float value = 0.f;
};

// Lightroom-style sliders (all neutral at 0). Applied as a left multiply on the
// inverse opsin 3×3 (output-side linear RGB), after any custom matrix already
// in the file. Units are chosen to be intuitive at moderate values.
struct OpsinAdjustParams {
  // Exposure in stops (2^stops scales the whole inverse matrix).
  float exposure_ev = 0.f;
  // White balance: warm (positive) boosts red vs blue output, roughly −100..100.
  float temperature = 0.f;
  // Tint: green (positive) vs magenta (negative), roughly −100..100.
  float tint = 0.f;
  // Hue −100..100 → h∈[−1,1] → ±90° rotation of decoder linear RGB about Rec. 709
  // luma (HSL/HSV-like; still one 3×3 left factor on the inverse opsin matrix).
  float hue = 0.f;
  // When false (default): left-multiply M by 2^ev · R(h) · D_tint(u) · D_temp(t)
  // (same order as libjxl-style sliders). When true: left-multiply by the exact
  // inverse of that transform using the same ev/temperature/tint/hue magnitudes
  // (used by --check_reversible undo; temperature/tint are not inverted by sign).
  bool undo_inverse = false;
  // Additive opsin bias tweak (XYB); range enforced by CLI (±0.003 on X/Y, ±0.03 on B).
  OpsinBiasChannelOpt opsin_bias_xyb[3]{};
  // Additive deltas on CustomTransformData quant_biases[0..3]; CLI range ±2 each.
  OpsinBiasChannelOpt quant_bias_delta[4]{};
};

// Returns false if the image is not XYB-encoded or parameters are invalid.
// When |did_change| is non-null, sets it to true iff the image header was
// updated (skipped when all parameters are neutral).
bool ApplyOpsinAdjust(ParsedCodestream* cs, const OpsinAdjustParams& params,
                      bool* did_change = nullptr);

// After a successful ApplyOpsinAdjust that changed opsin / quant biases, compute
// additive deltas that return those fields to implicit encoder defaults (F16-safe
// inverse for --check_reversible). |set[k]| selects channels that were adjusted
// on this run.
void ComputeOpsinBiasUndoToImplicitDefault(const ImageMetadata& meta,
                                           const bool set[3], float undo_delta[3]);
void ComputeQuantBiasUndoToImplicitDefault(const ImageMetadata& meta,
                                           const bool set[4], float undo_delta[4]);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_OPSIN_ADJUST_H_
