// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "operations.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "codestream.h"
#include "entropy.h"
#include "frame_header.h"
#include "photon_noise.h"
#include "printf_macros.h"
#include "lfglobal.h"
#include "orientation_compose.h"
#include "restoration_filter.h"
#include "spline_io.h"

namespace jxltran {

bool ApplyHeaderMod(ParsedCodestream* cs, const HeaderMod& mod) {
  ImageMetadata* m = &cs->image.metadata;
  if (mod.set_orientation != 0) {
    if (mod.set_orientation < 1 || mod.set_orientation > 8) {
      fprintf(stderr,
              "jxltran: --set-orientation value %u is out of range [1,8]\n",
              mod.set_orientation);
      return false;
    }
    m->orientation = mod.set_orientation;
    if (mod.set_orientation != 1) {
      m->all_default = false;
    }
  }

  if (mod.set_bits_per_sample != 0) {
    if (!m->xyb_encoded) {
      fprintf(stderr,
              "jxltran: --set-bits-per-sample requires xyb_encoded=true; "
              "this image is not XYB-encoded\n");
      return false;
    }
    m->bit_depth.bits_per_sample = mod.set_bits_per_sample;
    m->all_default = false;
  }

  if (mod.have_set_num_loops) {
    if (!m->have_animation) {
      fprintf(stderr,
              "jxltran: --set-num-loops requires an animated image "
              "(have_animation=true)\n");
      return false;
    }
    m->animation.num_loops = mod.set_num_loops;
  }

  if (mod.have_set_tps) {
    if (!m->have_animation) {
      fprintf(stderr,
              "jxltran: --set-tps requires an animated image "
              "(have_animation=true)\n");
      return false;
    }
    m->animation.tps_numerator = mod.set_tps_numerator;
    m->animation.tps_denominator = mod.set_tps_denominator;
  }

  return true;
}

// Output pixel (px,py) samples input pixel (px+dx, py+dy). Equivalently,
// input (ix,iy) appears at output (ix-dx, iy-dy). The codestream canvas size
// (SizeHeader) becomes new_w x new_h; Regular / SkipProgressive frame headers
// keep the same encoded dimensions (crop_width x crop_height, or full canvas
// when !have_crop) with updated origins. The decoder only composites the
// overlap with the canvas. LF and ReferenceOnly frame headers are left
// unchanged in memory; WriteCodestream emits those frames verbatim from the
// original codestream so their bit length and TOC stay aligned with payload.

static void ClearRedundantFullCanvasCrop(FrameHeader* fh, uint32_t canvas_w,
                                         uint32_t canvas_h) {
  if (!fh->have_crop) return;
  if (fh->frame_type == kFrameTypeLF ||
      fh->frame_type == kFrameTypeReferenceOnly) {
    return;
  }
  const int32_t x0 = UnpackSigned(fh->ux0);
  const int32_t y0 = UnpackSigned(fh->uy0);
  if (x0 == 0 && y0 == 0 && fh->crop_width == canvas_w &&
      fh->crop_height == canvas_h) {
    fh->have_crop = false;
    fh->ux0 = 0;
    fh->uy0 = 0;
    fh->crop_width = 0;
    fh->crop_height = 0;
  }
}

bool ApplyCrop(ParsedCodestream* cs, int32_t dx, int32_t dy, uint32_t new_w,
               uint32_t new_h) {
  const uint32_t old_w = cs->image.size.width;
  const uint32_t old_h = cs->image.size.height;

  if (new_w == 0 || new_h == 0) {
    fprintf(stderr, "jxltran: crop dimensions must be non-zero\n");
    return false;
  }

  if (new_w == old_w && new_h == old_h && dx == 0 && dy == 0) {
    return true;
  }

  cs->image.size.width = new_w;
  cs->image.size.height = new_h;

  for (FramedUnit& fu : cs->frames) {
    FrameHeader* fh = &fu.frame;
    const uint32_t ft = fh->frame_type;

    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }

    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) {
      continue;
    }

    int64_t x0;
    int64_t y0;
    uint32_t fw;
    uint32_t fhh;
    if (!fh->have_crop) {
      x0 = 0;
      y0 = 0;
      fw = old_w;
      fhh = old_h;
    } else {
      x0 = UnpackSigned(fh->ux0);
      y0 = UnpackSigned(fh->uy0);
      fw = fh->crop_width;
      fhh = fh->crop_height;
    }

    const int64_t nx = x0 - static_cast<int64_t>(dx);
    const int64_t ny = y0 - static_cast<int64_t>(dy);
    if (nx < std::numeric_limits<int32_t>::min() ||
        nx > std::numeric_limits<int32_t>::max() ||
        ny < std::numeric_limits<int32_t>::min() ||
        ny > std::numeric_limits<int32_t>::max()) {
      fprintf(stderr, "jxltran: crop: frame origin overflow\n");
      return false;
    }

    fh->all_default = false;
    fh->have_crop = true;
    fh->ux0 = PackSigned(static_cast<int32_t>(nx));
    fh->uy0 = PackSigned(static_cast<int32_t>(ny));
    fh->crop_width = fw;
    fh->crop_height = fhh;

    ClearRedundantFullCanvasCrop(fh, new_w, new_h);
  }

  return true;
}

bool ApplySetFrameRegionFromDisplay(ParsedCodestream* cs, size_t frame_index,
                                    uint32_t display_space_orientation,
                                    int32_t display_x, int32_t display_y,
                                    uint32_t display_w, uint32_t display_h,
                                    bool* out_changed) {
  if (out_changed) *out_changed = false;
  if (frame_index >= cs->frames.size()) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame index %" PRIuS " out of range\n",
            frame_index);
    return false;
  }
  FramedUnit& fu = cs->frames[frame_index];
  FrameHeader* fh = &fu.frame;
  const uint32_t ft = fh->frame_type;
  if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame %" PRIuS " cannot be repositioned "
            "(LF or reference-only)\n",
            frame_index);
    return false;
  }
  if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame %" PRIuS " has unsupported type\n",
            frame_index);
    return false;
  }
  uint32_t orient = display_space_orientation;
  if (orient < 1 || orient > 8) orient = 1;
  const uint32_t canvas_w = cs->image.size.width;
  const uint32_t canvas_h = cs->image.size.height;

  int64_t sx_old = 0;
  int64_t sy_old = 0;
  uint64_t rw = canvas_w;
  uint64_t rh = canvas_h;
  if (fh->have_crop) {
    sx_old = UnpackSigned(fh->ux0);
    sy_old = UnpackSigned(fh->uy0);
    rw = fh->crop_width;
    rh = fh->crop_height;
  }

  int64_t cur_dx = 0;
  int64_t cur_dy = 0;
  uint64_t cur_dw = 0;
  uint64_t cur_dh = 0;
  StorageRectToDisplayAabb(orient, canvas_w, canvas_h, sx_old, sy_old, rw, rh,
                           &cur_dx, &cur_dy, &cur_dw, &cur_dh);

  if (!fh->have_crop) {
    if (static_cast<uint64_t>(display_w) != cur_dw ||
        static_cast<uint64_t>(display_h) != cur_dh) {
      fprintf(stderr,
              "jxltran: --set-frame-region: frame %" PRIuS " covers the full canvas; "
              "only the current WxH+offsets are allowed (reposition is not "
              "supported)\n",
              frame_index);
      return false;
    }
    if (static_cast<int64_t>(display_x) == cur_dx &&
        static_cast<int64_t>(display_y) == cur_dy) {
      return true;
    }
    fprintf(stderr,
            "jxltran: --set-frame-region: frame %" PRIuS " covers the full canvas; "
            "cannot change origin\n",
            frame_index);
    return false;
  }

  if (static_cast<uint64_t>(display_w) != cur_dw ||
      static_cast<uint64_t>(display_h) != cur_dh) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame %" PRIuS " display size must stay "
            "%llu x %llu (got %u x %u)\n",
            frame_index,
            static_cast<unsigned long long>(cur_dw),
            static_cast<unsigned long long>(cur_dh),
            static_cast<unsigned>(display_w), static_cast<unsigned>(display_h));
    return false;
  }

  int32_t nsx = 0;
  int32_t nsy = 0;
  if (!DisplayAabbMinToStorageCropOrigin(
          orient, canvas_w, canvas_h, static_cast<int64_t>(display_x),
          static_cast<int64_t>(display_y), rw, rh, &nsx, &nsy)) {
    return false;
  }
  if (nsx == static_cast<int32_t>(sx_old) && nsy == static_cast<int32_t>(sy_old)) {
    return true;
  }

  if (out_changed) *out_changed = true;
  fh->all_default = false;
  fh->have_crop = true;
  fh->ux0 = PackSigned(nsx);
  fh->uy0 = PackSigned(nsy);
  fh->crop_width = static_cast<uint32_t>(rw);
  fh->crop_height = static_cast<uint32_t>(rh);
  ClearRedundantFullCanvasCrop(fh, canvas_w, canvas_h);
  return true;
}

namespace {

constexpr float kGabDef1 = 1.1f * 0.104699568f;
constexpr float kGabDef2 = 1.1f * 0.055680538f;
// Bounds on s = w1+w2 per XYB channel (decoder neighbor kernel shape). Default
// sum is kGabSumDef (~0.176). Sharpen/blur shift the same amount in logit(s)
// space so equal --gab-blur=A and --gab-sharpen=A cancel; sharpen amount 1
// from the default sum moves s halfway toward the min sum (max sharp).
constexpr float kGabSumMin = -0.4f;
constexpr float kGabSumMax = 2.0f;
constexpr float kGabSumSpan = kGabSumMax - kGabSumMin;
constexpr float kGabSumDef = kGabDef1 + kGabDef2;
constexpr float kGabLogitEpsilon = 1e-7f;
// Snap within this tolerance of implicit defaults to gab_custom=false (F16
// quantization + logit round-trip).
constexpr float kGabSnapToImplicitEps = 1.5e-4f;

bool GabPairOk(float w1, float w2) {
  return std::abs(1.0f + (w1 + w2) * 4.0f) >= 1e-8f;
}

bool GabRfOk(const ParsedRestorationFilter& rf) {
  if (!rf.gab || !rf.gab_custom) return true;
  return GabPairOk(rf.gab_x_weight1, rf.gab_x_weight2) &&
         GabPairOk(rf.gab_y_weight1, rf.gab_y_weight2) &&
         GabPairOk(rf.gab_b_weight1, rf.gab_b_weight2);
}

void SetStandardGab(ParsedRestorationFilter* rf, const float w1[3],
                    const float w2[3]) {
  rf->all_default = false;
  rf->gab = true;
  rf->gab_custom = true;
  rf->gab_x_weight1 = w1[0];
  rf->gab_x_weight2 = w2[0];
  rf->gab_y_weight1 = w1[1];
  rf->gab_y_weight2 = w2[1];
  rf->gab_b_weight1 = w1[2];
  rf->gab_b_weight2 = w2[2];
}

float GabClampSum(float s) {
  return std::min(std::max(s, kGabSumMin), kGabSumMax);
}

float GabSumToU(float s) {
  s = GabClampSum(s);
  float p = (s - kGabSumMin) / kGabSumSpan;
  p = std::min(std::max(p, kGabLogitEpsilon), 1.f - kGabLogitEpsilon);
  return std::log(p / (1.f - p));
}

float GabUToSum(float u) {
  const float p = 1.f / (1.f + std::exp(-u));
  return kGabSumMin + p * kGabSumSpan;
}

float GabUnitStepU() {
  static const float kStep = GabSumToU(kGabSumDef) -
                           GabSumToU((kGabSumDef + kGabSumMin) * 0.5f);
  return kStep;
}

void GabRescaleWeightsToSum(float w1, float w2, float s_new, float* o1,
                            float* o2) {
  const float s_old = w1 + w2;
  if (std::abs(s_old) < 1e-20f) {
    const float inv = 1.f / kGabSumDef;
    *o1 = kGabDef1 * inv * s_new;
    *o2 = kGabDef2 * inv * s_new;
    return;
  }
  const float scale = s_new / s_old;
  *o1 = w1 * scale;
  *o2 = w2 * scale;
}

// If all six weights match implicit encoder defaults within tolerance, store as
// gab on + gab_custom off (avoids F16 / logit drift breaking blur↔sharpen).
void MaybeSnapGabToImplicitDefault(ParsedRestorationFilter* rf) {
  if (!rf->gab || !rf->gab_custom) return;
  const auto near = [](float a, float b) {
    return std::abs(a - b) <= kGabSnapToImplicitEps;
  };
  if (near(rf->gab_x_weight1, kGabDef1) && near(rf->gab_x_weight2, kGabDef2) &&
      near(rf->gab_y_weight1, kGabDef1) && near(rf->gab_y_weight2, kGabDef2) &&
      near(rf->gab_b_weight1, kGabDef1) && near(rf->gab_b_weight2, kGabDef2)) {
    rf->gab_custom = false;
  }
}

// Decoder-effective neighbor weights: gab off → zeros; gab on implicit default
// → libjxl defaults (kGabDef*); custom → stored values.
void EffectiveGabWeights(const ParsedRestorationFilter& rf, float w1_out[3],
                         float w2_out[3]) {
  if (!rf.gab) {
    for (int c = 0; c < 3; ++c) {
      w1_out[c] = 0.f;
      w2_out[c] = 0.f;
    }
    return;
  }
  if (!rf.gab_custom) {
    for (int c = 0; c < 3; ++c) {
      w1_out[c] = kGabDef1;
      w2_out[c] = kGabDef2;
    }
    return;
  }
  w1_out[0] = rf.gab_x_weight1;
  w2_out[0] = rf.gab_x_weight2;
  w1_out[1] = rf.gab_y_weight1;
  w2_out[1] = rf.gab_y_weight2;
  w1_out[2] = rf.gab_b_weight1;
  w2_out[2] = rf.gab_b_weight2;
}

}  // namespace

bool ApplyGabArgs(ParsedCodestream* cs, const GabArgs& args,
                  const std::vector<size_t>* only_frames) {
  if (args.kind == GabArgs::Kind::kNone) return true;
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  size_t applied = 0;
  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->restoration.all_default) {
      // Only the all-default flag was read; EPF fields were never parsed.
      // Expand to explicit libjxl defaults (LoopFilter::SetDefault), then keep
      // Gaborish off until blur/sharpen applies weights (effective baseline 0).
      const bool modular = (fh->encoding == kFrameEncModular);
      fh->restoration = DefaultRestorationFilter(modular);
      fh->restoration.gab = false;
      fh->restoration.gab_custom = false;
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    if (args.kind == GabArgs::Kind::kCustom) {
      rf->all_default = false;
      rf->gab = true;
      rf->gab_custom = true;
      rf->gab_x_weight1 = args.custom[0];
      rf->gab_x_weight2 = args.custom[1];
      rf->gab_y_weight1 = args.custom[2];
      rf->gab_y_weight2 = args.custom[3];
      rf->gab_b_weight1 = args.custom[4];
      rf->gab_b_weight2 = args.custom[5];
    } else if (args.kind == GabArgs::Kind::kBlur ||
               args.kind == GabArgs::Kind::kSharpen) {
      float cur_w1[3], cur_w2[3];
      EffectiveGabWeights(*rf, cur_w1, cur_w2);
      const float step_u = GabUnitStepU();
      const float du =
          args.amount * step_u *
          ((args.kind == GabArgs::Kind::kBlur) ? 1.f : -1.f);
      float w1[3];
      float w2[3];
      for (int c = 0; c < 3; ++c) {
        const float s = cur_w1[c] + cur_w2[c];
        float u = GabSumToU(s);
        u += du;
        const float s_new = GabClampSum(GabUToSum(u));
        GabRescaleWeightsToSum(cur_w1[c], cur_w2[c], s_new, &w1[c], &w2[c]);
      }
      SetStandardGab(rf, w1, w2);
    }
    MaybeSnapGabToImplicitDefault(rf);
    if (!GabRfOk(*rf)) {
      fprintf(stderr,
              "jxltran: Gaborish kernel is near-singular: require "
              "|1+4*(weight1+weight2)| >= 1e-8 per XYB channel (same as libjxl "
              "decode). Reduce --gab-sharpen / --gab-blur or adjust "
              "--gab-weights.\n");
      return false;
    }
    fh->rf_reencode = true;
    ++applied;
  }
  if (applied == 0) {
    fprintf(stderr,
            "jxltran: no regular or skip-progressive frames to edit (only LF "
            "or reference-only frames).\n");
    return false;
  }
  return true;
}

bool ApplyEpfIters(ParsedCodestream* cs, uint32_t epf_iters,
                   const std::vector<size_t>* only_frames) {
  epf_iters &= 3u;
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  size_t applied = 0;
  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->restoration.all_default) {
      const bool modular = (fh->encoding == kFrameEncModular);
      fh->restoration = DefaultRestorationFilter(modular);
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    rf->all_default = false;
    rf->epf_iters = epf_iters;
    if (fh->encoding == kFrameEncModular && epf_iters > 0) {
      if (rf->epf_sigma_for_modular < 1e-8f) {
        rf->epf_sigma_for_modular = 1.f;
      }
    }
    fh->rf_reencode = true;
    ++applied;
  }
  if (applied == 0) {
    fprintf(stderr,
            "jxltran: no regular or skip-progressive frames to edit (only LF "
            "or reference-only frames).\n");
    return false;
  }
  return true;
}

// lib/jxl/loop_filter.cc VisitFields defaults for the EPF sigma bundle.
constexpr float kEpfQuantMulImplicit = 0.46f;
constexpr float kEpfSigmaModularImplicit = 1.0f;
constexpr float kEpfPass0SigmaDefault = 0.9f;
constexpr float kEpfPass2SigmaDefault = 6.5f;
constexpr float kEpfBorderSadDefault = 2.f / 3.f;
constexpr float kEpfAmpSnapEps = 1.5e-4f;

void MaybeSnapEpfAmplitudeDefaults(bool modular, ParsedRestorationFilter* rf) {
  if (rf->epf_iters == 0) return;
  if (modular) {
    if (std::abs(rf->epf_sigma_for_modular - kEpfSigmaModularImplicit) <=
        kEpfAmpSnapEps) {
      rf->epf_sigma_for_modular = kEpfSigmaModularImplicit;
    }
    if (!rf->epf_sigma_custom) return;
    const auto near = [](float a, float b) {
      return std::abs(a - b) <= kEpfAmpSnapEps;
    };
    if (near(rf->epf_pass0_sigma_scale, kEpfPass0SigmaDefault) &&
        near(rf->epf_pass2_sigma_scale, kEpfPass2SigmaDefault) &&
        near(rf->epf_border_sad_mul, kEpfBorderSadDefault) &&
        near(rf->epf_sigma_for_modular, kEpfSigmaModularImplicit)) {
      rf->epf_sigma_custom = false;
    }
    return;
  }
  if (!rf->epf_sigma_custom) return;
  const auto near = [](float a, float b) {
    return std::abs(a - b) <= kEpfAmpSnapEps;
  };
  if (near(rf->epf_quant_mul, kEpfQuantMulImplicit) &&
      near(rf->epf_pass0_sigma_scale, kEpfPass0SigmaDefault) &&
      near(rf->epf_pass2_sigma_scale, kEpfPass2SigmaDefault) &&
      near(rf->epf_border_sad_mul, kEpfBorderSadDefault)) {
    rf->epf_sigma_custom = false;
  }
}

bool ApplyEpfAmplitudeScale(ParsedCodestream* cs, float factor,
                            const std::vector<size_t>* only_frames) {
  if (!(factor > 0.f) || !std::isfinite(factor)) {
    fprintf(stderr,
            "jxltran: --set-epf-amplitude-scale expects a positive finite "
            "float\n");
    return false;
  }
  if (factor == 1.f) return true;

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  size_t applied = 0;
  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->restoration.all_default) {
      const bool modular = (fh->encoding == kFrameEncModular);
      fh->restoration = DefaultRestorationFilter(modular);
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    const uint32_t iters = std::min(3u, rf->epf_iters);
    if (iters == 0) continue;

    const bool modular = (fh->encoding == kFrameEncModular);
    rf->all_default = false;

    if (modular) {
      const float s = rf->epf_sigma_for_modular * factor;
      if (!(s > 1e-8f) || !std::isfinite(s)) {
        fprintf(stderr,
                "jxltran: --set-epf-amplitude-scale produced an invalid "
                "epf_sigma_for_modular for frame %" PRIuS "\n",
                fi);
        return false;
      }
      rf->epf_sigma_for_modular = s;
      MaybeSnapEpfAmplitudeDefaults(true, rf);
    } else {
      const float base =
          rf->epf_sigma_custom ? rf->epf_quant_mul : kEpfQuantMulImplicit;
      const float new_q = base * factor;
      if (!(new_q > 0.f) || !std::isfinite(new_q)) {
        fprintf(stderr,
                "jxltran: --set-epf-amplitude-scale produced an invalid "
                "epf_quant_mul for frame %" PRIuS "\n",
                fi);
        return false;
      }
      if (!rf->epf_sigma_custom) {
        rf->epf_pass0_sigma_scale = kEpfPass0SigmaDefault;
        rf->epf_pass2_sigma_scale = kEpfPass2SigmaDefault;
        rf->epf_border_sad_mul = kEpfBorderSadDefault;
      }
      rf->epf_sigma_custom = true;
      rf->epf_quant_mul = new_q;
      MaybeSnapEpfAmplitudeDefaults(false, rf);
    }

    fh->rf_reencode = true;
    ++applied;
  }
  if (applied == 0) {
    fprintf(stderr,
            "jxltran: --set-epf-amplitude-scale: no frames with EPF enabled "
            "(epf_iters==0 on all edited frames, or only LF / "
            "reference-only).\n");
    return false;
  }
  return true;
}

bool EpfEffectiveAmplitudeForInfo(bool modular_encoding,
                                  bool restoration_all_default,
                                  const ParsedRestorationFilter& rf,
                                  float* out_amp) {
  ParsedRestorationFilter eff = rf;
  if (restoration_all_default) {
    eff = DefaultRestorationFilter(modular_encoding);
  }
  const uint32_t it = std::min(3u, eff.epf_iters);
  if (it == 0) return false;
  if (modular_encoding) {
    *out_amp = eff.epf_sigma_for_modular;
  } else if (!eff.epf_sigma_custom) {
    *out_amp = kEpfQuantMulImplicit;
  } else {
    *out_amp = eff.epf_quant_mul;
  }
  return true;
}

static void FillEpfSharpLutMixed(float uniformity,
                                 std::array<float, kEpfSharpEntries>* out) {
  for (int i = 0; i < kEpfSharpEntries; ++i) {
    const float d = static_cast<float>(i) / 7.f;
    (*out)[static_cast<size_t>(i)] = (1.f - uniformity) * d + uniformity;
  }
}

static void MaybeSnapEpfSharpLutToImplicitRamp(ParsedRestorationFilter* rf) {
  if (!rf->epf_sharp_custom || rf->epf_iters == 0) return;
  for (int i = 0; i < kEpfSharpEntries; ++i) {
    const float d = static_cast<float>(i) / 7.f;
    if (std::abs(rf->epf_sharp_lut[static_cast<size_t>(i)] - d) > 1.5e-4f) {
      return;
    }
  }
  rf->epf_sharp_custom = false;
}

bool ApplyEpfSharpUniformity(ParsedCodestream* cs, float uniformity,
                             const std::vector<size_t>* only_frames,
                             bool* out_changed) {
  bool changed = false;
  if (out_changed != nullptr) *out_changed = false;
  if (!std::isfinite(uniformity) || uniformity < 0.f || uniformity > 1.f) {
    fprintf(stderr,
            "jxltran: --set-epf-uniformity expects a float in [0,1]\n");
    return false;
  }
  constexpr float kEps = 1e-6f;
  float u = uniformity;
  if (u <= kEps) u = 0.f;
  if (u + kEps >= 1.f) u = 1.f;

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  size_t eligible = 0;
  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->encoding == kFrameEncModular) {
      continue;
    }
    if (fh->restoration.all_default) {
      fh->restoration = DefaultRestorationFilter(false);
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    const uint32_t iters = std::min(3u, rf->epf_iters);
    if (iters == 0) continue;
    ++eligible;

    rf->all_default = false;
    if (u == 0.f) {
      if (rf->epf_sharp_custom) {
        rf->epf_sharp_custom = false;
        fh->rf_reencode = true;
        changed = true;
      }
    } else {
      rf->epf_sharp_custom = true;
      FillEpfSharpLutMixed(u, &rf->epf_sharp_lut);
      MaybeSnapEpfSharpLutToImplicitRamp(rf);
      fh->rf_reencode = true;
      changed = true;
    }
  }

  if (out_changed != nullptr) *out_changed = changed;

  if (u > 0.f && eligible == 0) {
    fprintf(stderr,
            "jxltran: --set-epf-uniformity: no VarDCT frames with EPF "
            "(epf_iters==0 on all edited VarDCT frames, or only modular / LF / "
            "reference-only).\n");
    return false;
  }
  return true;
}

float EpfSharpUniformityForInfo(bool restoration_all_default,
                                const ParsedRestorationFilter& rf) {
  ParsedRestorationFilter eff = rf;
  if (restoration_all_default) {
    eff = DefaultRestorationFilter(false);
  }
  if (eff.epf_iters == 0) return 0.f;
  if (!eff.epf_sharp_custom) return 0.f;
  const float t = eff.epf_sharp_lut[0];
  if (t < 0.f) return 0.f;
  if (t > 1.f) return 1.f;
  return t;
}

namespace {

bool LogicalSection0BodyByteOffset(const FramedUnit& fu, size_t* out_bytes) {
  if (fu.toc_decoded_sizes.empty()) return false;
  if (fu.toc_perm.empty()) {
    *out_bytes = 0;
    return true;
  }
  for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
    if (fu.toc_perm[p] == 0) {
      size_t off = 0;
      for (size_t q = 0; q < p; ++q) {
        if (q >= fu.toc_decoded_sizes.size()) return false;
        off += fu.toc_decoded_sizes[q];
      }
      *out_bytes = off;
      return true;
    }
  }
  return false;
}

// Stream-order TOC slot s carries logical section perm[s]; lf_global is logical 0.
static bool PhysicalStreamIndexForLogical0(const FramedUnit& fu, size_t* p0) {
  if (fu.toc_decoded_sizes.empty()) return false;
  if (fu.toc_perm.empty()) {
    *p0 = 0;
    return true;
  }
  for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
    if (fu.toc_perm[p] == 0) {
      *p0 = p;
      return true;
    }
  }
  return false;
}

static bool LfGlobalSection0BitLength(const FramedUnit& fu, size_t* bits) {
  if (fu.toc_decoded_sizes.empty()) return false;
  size_t p0 = 0;
  if (!PhysicalStreamIndexForLogical0(fu, &p0)) return false;
  *bits = static_cast<size_t>(fu.toc_decoded_sizes[p0]) * 8u;
  return true;
}

}  // namespace

bool ApplyPhotonNoiseIso(ParsedCodestream* cs, bool change, float iso,
                         const std::vector<size_t>* only_frames) {
  if (!change) return true;

  const uint32_t canvas_w = cs->image.size.width;
  const uint32_t canvas_h = cs->image.size.height;

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  for (size_t fi = 0; fi < cs->frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    FramedUnit& fu = cs->frames[fi];
    if (fu.spline_edit) {
      fprintf(stderr,
              "jxltran: --set-photon-noise-iso cannot be combined with "
              "--set-splines-from on the same output pass (frame %" PRIuS ").\n",
              fi);
      return false;
    }
    fu.photon_noise_edit = false;
    fu.photon_noise_delta_bytes = 0;
    fu.photon_noise_patch_abs_bit = 0;
    fu.photon_noise_new_bytes.fill(0);
    fu.photon_noise_toc_reencode = false;

    FrameHeader* fh = &fu.frame;
    const uint32_t ft = fh->frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) {
      continue;
    }

    size_t sec0_bytes = 0;
    if (!LogicalSection0BodyByteOffset(fu, &sec0_bytes)) {
      fprintf(stderr, "jxltran: photon noise: invalid TOC permutation\n");
      return false;
    }

    const size_t xsize_px = fh->have_crop ? fh->crop_width : canvas_w;
    const size_t ysize_px = fh->have_crop ? fh->crop_height : canvas_h;
    if (xsize_px == 0 || ysize_px == 0) {
      fprintf(stderr, "jxltran: photon noise: invalid frame dimensions\n");
      return false;
    }

    const size_t frame_bit0 = fu.original_frame_byte_offset * 8;
    const size_t body0 = frame_bit0 + fu.toc_start_bit + fu.toc_bit_length;
    const size_t lf_global_abs = body0 + sec0_bytes * 8;

    size_t patch_abs = lf_global_abs;
    if (fu.lf_global_noise_lut_abs_valid) {
      patch_abs = fu.lf_global_noise_lut_abs_bit;
    } else if ((fh->flags & kFrameFlagPatches) != 0 ||
               (fh->flags & kFrameFlagSplines) != 0) {
      fprintf(stderr,
              "jxltran: --set-photon-noise-iso: could not parse DC-global prefix "
              "(patches/splines); noise LUT position is unknown.\n");
      return false;
    }

    const bool had_noise = (fh->flags & kFrameFlagNoise) != 0;

    size_t s0_bits = 0;
    if (!LfGlobalSection0BitLength(fu, &s0_bits)) {
      fprintf(stderr, "jxltran: photon noise: missing TOC sizes for DC global\n");
      return false;
    }
    const size_t noise_rel =
        fu.lf_global_noise_lut_abs_valid ? fu.lf_global_noise_lut_rel_bit : 0;

    if (iso <= 0.f) {
      if (!had_noise) {
        continue;
      }
      if (noise_rel + 80 > s0_bits) {
        fprintf(stderr,
                "jxltran: photon noise: noise LUT extends past DC-global section\n");
        return false;
      }
      fh->flags &= ~kFrameFlagNoise;
      fh->all_default = false;
      fu.photon_noise_edit = true;
      fu.photon_noise_delta_bytes = -10;
      fu.photon_noise_patch_abs_bit = patch_abs;
      fu.photon_noise_toc_reencode = true;
      if (!fu.toc_decoded_sizes.empty()) {
        size_t p0 = 0;
        if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
          fprintf(stderr, "jxltran: photon noise: invalid TOC permutation\n");
          return false;
        }
        const uint32_t s0 = fu.toc_decoded_sizes[p0];
        if (s0 < 10u) {
          fprintf(stderr,
                  "jxltran: photon noise: DC global section too small to remove "
                  "noise\n");
          return false;
        }
        fu.toc_decoded_sizes[p0] = s0 - 10u;
      }
      fu.body_bit_length -= 80;
      continue;
    }

    std::array<float, kNoiseLutPoints> lut{};
    SimulatePhotonNoise(xsize_px, ysize_px, iso, &lut);
    if (!NoiseLutHasAny(lut)) {
      if (!had_noise) {
        continue;
      }
      if (noise_rel + 80 > s0_bits) {
        fprintf(stderr,
                "jxltran: photon noise: noise LUT extends past DC-global section\n");
        return false;
      }
      fh->flags &= ~kFrameFlagNoise;
      fh->all_default = false;
      fu.photon_noise_edit = true;
      fu.photon_noise_delta_bytes = -10;
      fu.photon_noise_patch_abs_bit = patch_abs;
      fu.photon_noise_toc_reencode = true;
      if (!fu.toc_decoded_sizes.empty()) {
        size_t p0 = 0;
        if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
          fprintf(stderr, "jxltran: photon noise: invalid TOC permutation\n");
          return false;
        }
        const uint32_t s0 = fu.toc_decoded_sizes[p0];
        if (s0 < 10u) {
          fprintf(stderr,
                  "jxltran: photon noise: DC global section too small to remove "
                  "noise\n");
          return false;
        }
        fu.toc_decoded_sizes[p0] = s0 - 10u;
      }
      fu.body_bit_length -= 80;
      continue;
    }

    EncodeNoiseLutBits(lut, &fu.photon_noise_new_bytes);

    if (had_noise) {
      if (noise_rel + 80 > s0_bits) {
        fprintf(stderr,
                "jxltran: photon noise: noise LUT extends past DC-global section\n");
        return false;
      }
      fh->flags |= kFrameFlagNoise;
      fh->all_default = false;
      fu.photon_noise_edit = true;
      fu.photon_noise_delta_bytes = 0;
      fu.photon_noise_patch_abs_bit = patch_abs;
      fu.photon_noise_toc_reencode = false;
      continue;
    }

    if (noise_rel > s0_bits) {
      fprintf(stderr,
              "jxltran: photon noise: noise insert point past DC-global section\n");
      return false;
    }

    fh->flags |= kFrameFlagNoise;
    fh->all_default = false;
    fu.photon_noise_edit = true;
    fu.photon_noise_delta_bytes = 10;
    fu.photon_noise_patch_abs_bit = patch_abs;
    fu.photon_noise_toc_reencode = true;
    if (!fu.toc_decoded_sizes.empty()) {
      size_t p0 = 0;
      if (!PhysicalStreamIndexForLogical0(fu, &p0)) {
        fprintf(stderr, "jxltran: photon noise: invalid TOC permutation\n");
        return false;
      }
      fu.toc_decoded_sizes[p0] += 10u;
    }
    fu.body_bit_length += 80;
  }

  return true;
}

static bool ReadPathToString(const char* path, std::string* out) {
  FILE* f = fopen(path, "rb");
  if (f == nullptr) {
    fprintf(stderr, "jxltran: could not open '%s'\n", path);
    return false;
  }
  if (fseek(f, 0, SEEK_END) != 0) {
    fclose(f);
    return false;
  }
  const long sz = ftell(f);
  if (sz < 0) {
    fclose(f);
    return false;
  }
  if (fseek(f, 0, SEEK_SET) != 0) {
    fclose(f);
    return false;
  }
  out->assign(static_cast<size_t>(sz), '\0');
  if (sz > 0 &&
      fread(out->data(), 1, static_cast<size_t>(sz), f) !=
          static_cast<size_t>(sz)) {
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

bool ApplySplinesFromFile(ParsedCodestream* cs, const char* path,
                          const std::vector<size_t>* only_frames) {
  std::string text;
  if (!ReadPathToString(path, &text)) {
    return false;
  }
  std::vector<std::pair<size_t, LfGlobalSplines>> blocks;
  if (!SplinesParseText(text, &blocks)) {
    fprintf(stderr, "jxltran: failed to parse splines text file '%s'\n", path);
    return false;
  }

  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };

  size_t applied = 0;
  for (const auto& blk : blocks) {
    const size_t fi = blk.first;
    if (!want_frame(fi)) {
      continue;
    }
    if (fi >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --set-splines-from: frame %" PRIuS " is out of range\n",
              fi);
      return false;
    }
    FramedUnit& fu = cs->frames[fi];
    if (fu.photon_noise_edit) {
      fprintf(stderr,
              "jxltran: --set-splines-from cannot be combined with photon noise "
              "edit on the same pass (frame %" PRIuS ").\n",
              fi);
      return false;
    }
    if (!fu.lf_global_spline_region_valid) {
      fprintf(stderr,
              "jxltran: --set-splines-from: spline entropy insert/replace span "
              "unknown for frame %" PRIuS " (LF-global prefix not parsed)\n",
              fi);
      return false;
    }
    const bool had_splines = (fu.frame.flags & kFrameFlagSplines) != 0;
    if (had_splines) {
      if (fu.lf_global_spline_region_abs_start_bit >=
          fu.lf_global_spline_region_abs_end_bit) {
        fprintf(stderr,
                "jxltran: --set-splines-from: frame %" PRIuS " has invalid spline "
                "entropy span\n",
                fi);
        return false;
      }
    } else {
      if (fu.lf_global_spline_region_abs_start_bit !=
          fu.lf_global_spline_region_abs_end_bit) {
        fprintf(stderr,
                "jxltran: --set-splines-from: frame %" PRIuS " spline insert point "
                "is inconsistent\n",
                fi);
        return false;
      }
      fu.frame.flags |= kFrameFlagSplines;
    }
    if (fu.toc_strip_perm_reorder) {
      fprintf(stderr,
              "jxltran: --set-splines-from: incompatible with stripped TOC "
              "permutation (frame %" PRIuS ")\n",
              fi);
      return false;
    }

    const uint32_t canvas_w = cs->image.size.width;
    const uint32_t canvas_h = cs->image.size.height;
    FrameTocMetrics m{};
    ComputeFrameTocMetrics(fu.frame, cs->image.metadata, canvas_w, canvas_h,
                          &m);
    const size_t num_pixels = m.xsize * m.ysize;
    LfGlobalSplines spl = blk.second;
    std::vector<uint8_t> enc;
    size_t enc_bits = 0;
    if (!EncodeSplinesBundleBits(spl, num_pixels, &enc, &enc_bits,
                                 /*pad_entropy_to_byte=*/false)) {
      fprintf(stderr,
              "jxltran: --set-splines-from: failed to re-encode splines for frame "
              "%" PRIuS " (hybrid-uint / entropy re-encoding failed)\n",
              fi);
      return false;
    }

    const int64_t old_span_bits = static_cast<int64_t>(
        fu.lf_global_spline_region_abs_end_bit -
        fu.lf_global_spline_region_abs_start_bit);
    const int64_t new_span_bits = static_cast<int64_t>(enc_bits);
    const int64_t delta_content = new_span_bits - old_span_bits;
    const int64_t old_body_bits =
        static_cast<int64_t>(fu.body_bit_length);
    const int64_t new_body_core = old_body_bits + delta_content;
    if (new_body_core < 0) {
      fprintf(stderr,
              "jxltran: --set-splines-from: frame %" PRIuS " spline edit would "
              "make negative body length\n",
              fi);
      return false;
    }
    // Byte-align the whole frame body: pad with 0..7 zero bits. We do not parse
    // the entire LF-global bundle, so trailing bits in the last body byte may
    // already have been encoder padding—we cannot subtract those. Worst case we
    // add ~one redundant byte on top of what a full decoder would need.
    int64_t phase_mod = new_body_core % 8;
    if (phase_mod < 0) {
      phase_mod += 8;
    }
    const int64_t body_tail_pad = (8 - phase_mod) % 8;
    const int64_t delta_bits = delta_content + body_tail_pad;
    fu.spline_edit = true;
    fu.spline_edit_new_mid_bytes = std::move(enc);
    fu.spline_edit_new_mid_bits = enc_bits;
    fu.spline_edit_delta_bits = delta_bits;
    fu.spline_edit_toc_reencode =
        !fu.toc_perm.empty() && fu.spline_edit_delta_bits != 0;
    fu.lf_global_splines = std::move(spl);

    fu.body_bit_length = static_cast<size_t>(
        static_cast<int64_t>(fu.body_bit_length) + delta_bits);
    if (!fu.toc_decoded_sizes.empty()) {
      size_t p0 = 0;
      if (fu.toc_perm.empty()) {
        p0 = 0;
      } else {
        bool found = false;
        for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
          if (fu.toc_perm[p] == 0) {
            p0 = p;
            found = true;
            break;
          }
        }
        if (!found) {
          fprintf(stderr, "jxltran: --set-splines-from: invalid TOC permutation\n");
          return false;
        }
      }
      const int64_t dbytes = delta_bits / 8;
      const int64_t ns0 =
          static_cast<int64_t>(fu.toc_decoded_sizes[p0]) + dbytes;
      if (ns0 < 0 || ns0 > UINT32_MAX) {
        fprintf(stderr,
                "jxltran: --set-splines-from: DC global section size invalid\n");
        return false;
      }
      fu.toc_decoded_sizes[p0] = static_cast<uint32_t>(ns0);
    }
    ++applied;
  }

  if (applied == 0) {
    fprintf(stderr,
            "jxltran: --set-splines-from: no spline bundles applied (empty "
            "file, parse error, or no matching frame indices)\n");
    return false;
  }
  return true;
}

void GabEffectiveWeights6(bool modular_encoding, bool restoration_all_default,
                          const ParsedRestorationFilter& rf, float out6[6]) {
  ParsedRestorationFilter eff = rf;
  if (restoration_all_default) {
    eff = DefaultRestorationFilter(modular_encoding);
  }
  float w1[3];
  float w2[3];
  EffectiveGabWeights(eff, w1, w2);
  out6[0] = w1[0];
  out6[1] = w2[0];
  out6[2] = w1[1];
  out6[3] = w2[1];
  out6[4] = w1[2];
  out6[5] = w2[2];
}

namespace {

void NormalizeBlendingInfoForMode(uint32_t mode, uint32_t num_extra,
                                  bool is_partial_frame, FrameBlendingInfo* bi) {
  bi->mode = mode;
  const bool has_alpha = num_extra > 0 && (mode == 2u || mode == 3u);
  const bool has_clamp =
      num_extra > 0 && (mode == 2u || mode == 3u || mode == 4u);
  if (!has_alpha) bi->alpha_channel = 0;
  if (!has_clamp) bi->clamp = false;
  if (mode == 0u && !is_partial_frame) {
    bi->source = 0;
  } else {
    bi->source &= 3u;
  }
}

}  // namespace

bool ApplyFrameBlendOverrides(ParsedCodestream* cs,
                              const std::vector<FrameBlendOverride>& overrides) {
  const uint32_t cw = cs->image.size.width;
  const uint32_t ch = cs->image.size.height;
  const uint32_t nx = cs->image.metadata.num_extra;
  for (const FrameBlendOverride& o : overrides) {
    if (o.frame_index >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: frame index %" PRIuS " out of range "
              "(codestream has %" PRIuS " frames)\n",
              o.frame_index, cs->frames.size());
      return false;
    }
    if (o.mode > 4u) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: blend mode %u invalid "
              "(expect 0–4)\n",
              static_cast<unsigned>(o.mode));
      return false;
    }
    FrameHeader& fh = cs->frames[o.frame_index].frame;
    if (fh.frame_type != kFrameTypeRegular &&
        fh.frame_type != kFrameTypeSkipProgressive) {
      fprintf(stderr,
              "jxltran: --set-frame-blends: frame %" PRIuS " is not "
              "regular/skip-progressive\n",
              o.frame_index);
      return false;
    }
    const bool partial = FrameNeedsBlendingSourceField(fh, cw, ch);

    NormalizeBlendingInfoForMode(o.mode, nx, partial, &fh.blending_info);
    if (fh.ec_blending_info.size() != nx) {
      fh.ec_blending_info.assign(nx, FrameBlendingInfo{});
    }
    for (size_t e = 0; e < nx; ++e) {
      NormalizeBlendingInfoForMode(o.mode, nx, partial, &fh.ec_blending_info[e]);
    }

    auto apply_one = [&](FrameBlendingInfo* bi) {
      if (o.set_alpha_channel) {
        bi->alpha_channel = o.alpha_channel;
      }
      if (o.set_clamp) {
        bi->clamp = o.clamp;
      }
      if (o.set_source) {
        bi->source = o.source;
      }
      return true;
    };

    if (!apply_one(&fh.blending_info)) return false;
    for (size_t e = 0; e < nx; ++e) {
      if (!apply_one(&fh.ec_blending_info[e])) return false;
    }

    if (o.set_save_as_reference) {
      if (fh.is_last) {
        fprintf(stderr,
                "jxltran: --set-frame-blends: target= applies only when the "
                "frame is not the last in the codestream (is_last is false)\n");
        return false;
      }
      if (o.save_as_reference > 3u) {
        fprintf(stderr,
                "jxltran: --set-frame-blends: target= must be in range 0–3\n");
        return false;
      }
      fh.save_as_reference = o.save_as_reference & 3u;
    }
  }
  return true;
}

bool ApplyFrameDurationOverrides(
    ParsedCodestream* cs,
    const std::vector<std::pair<size_t, uint32_t>>& frame_index_duration) {
  if (!cs->image.metadata.have_animation) {
    fprintf(stderr,
            "jxltran: --set-frame-durations requires an animated image "
            "(have_animation)\n");
    return false;
  }
  for (const auto& p : frame_index_duration) {
    if (p.first >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-durations: frame index %" PRIuS " out of "
              "range (codestream has %" PRIuS " frames)\n",
              p.first, cs->frames.size());
      return false;
    }
    FrameHeader& fh = cs->frames[p.first].frame;
    if (fh.frame_type != kFrameTypeRegular &&
        fh.frame_type != kFrameTypeSkipProgressive) {
      fprintf(stderr,
              "jxltran: --set-frame-durations: frame %" PRIuS " is not "
              "regular/skip-progressive\n",
              p.first);
      return false;
    }
    fh.duration = p.second;
  }
  return true;
}

namespace {

constexpr size_t kMaxFrameNameBytes = 4096;

}  // namespace

bool ApplySetFrameNames(
    ParsedCodestream* cs,
    const std::vector<std::pair<size_t, std::vector<uint8_t>>>&
        frame_index_name) {
  for (const auto& p : frame_index_name) {
    if (p.first >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --set-frame-names: frame index %" PRIuS " out of range "
              "(codestream has %" PRIuS " frames)\n",
              p.first, cs->frames.size());
      return false;
    }
    FrameHeader& fh = cs->frames[p.first].frame;
    if (fh.frame_type != kFrameTypeRegular &&
        fh.frame_type != kFrameTypeSkipProgressive) {
      fprintf(stderr,
              "jxltran: --set-frame-names: frame %" PRIuS " is not "
              "regular/skip-progressive\n",
              p.first);
      return false;
    }
    if (p.second.size() > kMaxFrameNameBytes) {
      fprintf(stderr,
              "jxltran: --set-frame-names: name for frame %" PRIuS " exceeds "
              "%" PRIuS " bytes\n",
              p.first, kMaxFrameNameBytes);
      return false;
    }
    fh.name = p.second;
  }
  return true;
}

bool ApplyKeepListedFrames(ParsedCodestream* cs,
                           const std::vector<size_t>& frames_in_order,
                           bool* out_changed) {
  if (out_changed) *out_changed = false;
  if (frames_in_order.empty()) {
    fprintf(stderr,
            "jxltran: --keep-listed-frames requires a non-empty --keep-frames or "
            "--frames list\n");
    return false;
  }
  std::vector<size_t> order;
  order.reserve(frames_in_order.size());
  for (size_t idx : frames_in_order) {
    if (idx >= cs->frames.size()) {
      fprintf(stderr,
              "jxltran: --frames/--keep-frames: index %" PRIuS " out of range "
              "(codestream has "
              "%" PRIuS " frames)\n",
              idx, cs->frames.size());
      return false;
    }
    bool seen = false;
    for (size_t prior : order) {
      if (prior == idx) {
        seen = true;
        break;
      }
    }
    if (!seen) order.push_back(idx);
  }

  bool identity_ordered = order.size() == cs->frames.size();
  if (identity_ordered) {
    for (size_t i = 0; i < order.size(); ++i) {
      if (order[i] != i) {
        identity_ordered = false;
        break;
      }
    }
  }

  std::vector<std::pair<bool, uint32_t>> tail_before;
  tail_before.reserve(cs->frames.size());
  for (const FramedUnit& fu : cs->frames) {
    const FrameHeader& fh = fu.frame;
    if (fh.frame_type == kFrameTypeLF ||
        fh.frame_type == kFrameTypeReferenceOnly) {
      tail_before.emplace_back(false, 0);
    } else {
      tail_before.emplace_back(fh.is_last, fh.save_as_reference);
    }
  }

  std::vector<FramedUnit> src = std::move(cs->frames);
  std::vector<FramedUnit> out;
  out.reserve(order.size());
  for (size_t idx : order) {
    out.push_back(std::move(src[idx]));
  }
  cs->frames = std::move(out);

  const uint32_t last_ft = cs->frames.back().frame.frame_type;
  if (last_ft != kFrameTypeRegular && last_ft != kFrameTypeSkipProgressive) {
    fprintf(stderr,
            "jxltran: --keep-listed-frames: the last kept frame must be "
            "regular or skip-progressive (got frame_type=%u)\n",
            static_cast<unsigned>(last_ft));
    return false;
  }

  const uint32_t canvas_w = cs->image.size.width;
  const uint32_t canvas_h = cs->image.size.height;

  for (size_t i = 0; i < cs->frames.size(); ++i) {
    FrameHeader& fh = cs->frames[i].frame;
    if (fh.frame_type == kFrameTypeLF ||
        fh.frame_type == kFrameTypeReferenceOnly) {
      continue;
    }
    const bool last = (i + 1 == cs->frames.size());
    fh.is_last = last;
    if (last) {
      fh.save_as_reference = 0;
      continue;
    }
    const FrameHeader& nfh = cs->frames[i + 1].frame;
    uint32_t slot = fh.save_as_reference & 3u;
    if (nfh.frame_type == kFrameTypeRegular ||
        nfh.frame_type == kFrameTypeSkipProgressive) {
      const bool partial_next =
          FrameNeedsBlendingSourceField(nfh, canvas_w, canvas_h);
      if (nfh.blending_info.mode != 0u || partial_next) {
        slot = nfh.blending_info.source & 3u;
      }
    }
    fh.save_as_reference = slot;
    if (cs->image.metadata.have_animation && fh.duration != 0 &&
        fh.save_as_reference == 0) {
      fh.save_as_reference = 1u;
    }
  }

  if (out_changed) {
    if (!identity_ordered) {
      *out_changed = true;
    } else {
      bool diff = false;
      for (size_t j = 0; j < cs->frames.size(); ++j) {
        const FrameHeader& fh = cs->frames[j].frame;
        if (fh.frame_type == kFrameTypeLF ||
            fh.frame_type == kFrameTypeReferenceOnly) {
          continue;
        }
        const size_t src_i = order[j];
        if (src_i >= tail_before.size()) {
          diff = true;
          break;
        }
        if (fh.is_last != tail_before[src_i].first ||
            fh.save_as_reference != tail_before[src_i].second) {
          diff = true;
          break;
        }
      }
      *out_changed = diff;
    }
  }
  return true;
}

namespace {

bool ImageHeadersAppendCompatible(const ImageHeader& a, const ImageHeader& b,
                                  std::string* err) {
  const ImageMetadata& ma = a.metadata;
  const ImageMetadata& mb = b.metadata;
  if (ma.num_extra != mb.num_extra) {
    *err = "extra channel count mismatch";
    return false;
  }
  if (ma.xyb_encoded != mb.xyb_encoded) {
    *err = "xyb_encoded mismatch (need both XYB or both non-XYB)";
    return false;
  }
  if (!ma.xyb_encoded) {
    if (ma.bit_depth.bits_per_sample != mb.bit_depth.bits_per_sample ||
        ma.bit_depth.float_sample != mb.bit_depth.float_sample ||
        ma.bit_depth.exp_bits != mb.bit_depth.exp_bits) {
      *err = "main bit_depth mismatch (required when not XYB)";
      return false;
    }
  }
  return true;
}

// When the merged image header canvas is larger than |file_canvas| in either
// dimension, regular / skip-progressive frames that still use implicit
// full-canvas framing (!have_crop) must get an explicit crop so the encoded
// extent stays WxH at origin (0,0) on the larger canvas.
static void AppendMaterializeDefaultFrameCrops(std::vector<FramedUnit>* frames,
                                               uint32_t file_canvas_w,
                                               uint32_t file_canvas_h,
                                               uint32_t merged_w,
                                               uint32_t merged_h) {
  if (file_canvas_w >= merged_w && file_canvas_h >= merged_h) return;
  for (FramedUnit& fu : *frames) {
    FrameHeader* fh = &fu.frame;
    if (fh->frame_type != kFrameTypeRegular &&
        fh->frame_type != kFrameTypeSkipProgressive) {
      continue;
    }
    if (fh->have_crop) continue;
    fh->have_crop = true;
    fh->ux0 = PackSigned(0);
    fh->uy0 = PackSigned(0);
    fh->crop_width = file_canvas_w;
    fh->crop_height = file_canvas_h;
  }
}

void ShiftFramedUnitForAppend(FramedUnit* fu, int64_t byte_delta) {
  if (byte_delta < 0) {
    return;
  }
  const size_t bit_delta = static_cast<size_t>(byte_delta) * 8u;
  fu->original_frame_byte_offset =
      static_cast<size_t>(static_cast<int64_t>(fu->original_frame_byte_offset) +
                          byte_delta);
  if (fu->lf_global_noise_lut_abs_valid) {
    fu->lf_global_noise_lut_abs_bit += bit_delta;
  }
  if (fu->lf_global_spline_region_valid) {
    fu->lf_global_spline_region_abs_start_bit += bit_delta;
    fu->lf_global_spline_region_abs_end_bit += bit_delta;
  }
  if (fu->photon_noise_edit) {
    fu->photon_noise_patch_abs_bit += bit_delta;
  }
}

}  // namespace

// 42×3 modular zeros, single group; bare codestream. Base64 "/woQAFIASAgGAQAMAEsgGA==".
static constexpr uint8_t kBuiltinDummyTailCodestream[] = {
    0xff, 0x0a, 0x10, 0x00, 0x52, 0x00, 0x48, 0x08,
    0x06, 0x01, 0x00, 0x0c, 0x00, 0x4b, 0x20, 0x18,
};

static bool IsBuiltinDummyTailCodestreamBytes(const std::vector<uint8_t>& cs) {
  return cs.size() == sizeof(kBuiltinDummyTailCodestream) &&
         std::memcmp(cs.data(), kBuiltinDummyTailCodestream,
                     sizeof(kBuiltinDummyTailCodestream)) == 0;
}

std::vector<uint8_t> BuiltinAppendDummyTailCodestream() {
  return std::vector<uint8_t>(
      kBuiltinDummyTailCodestream,
      kBuiltinDummyTailCodestream + sizeof(kBuiltinDummyTailCodestream));
}

bool AppendCodestreamMerge(const std::vector<uint8_t>& primary_cs,
                            const std::vector<uint8_t>& append_cs,
                            std::vector<uint8_t>* merged_out, std::string* err,
                            bool skip_header_compat) {
  merged_out->clear();
  ParsedCodestream a;
  ParsedCodestream b;
  if (!ReadCodestream(primary_cs.data(), primary_cs.size(), &a)) {
    *err = "failed to parse primary codestream";
    return false;
  }
  if (!ReadCodestream(append_cs.data(), append_cs.size(), &b)) {
    *err = "failed to parse append codestream";
    return false;
  }
  if (a.frames.empty() || b.frames.empty()) {
    *err = "empty frame list";
    return false;
  }
  if (!skip_header_compat) {
    if (!ImageHeadersAppendCompatible(a.image, b.image, err)) {
      return false;
    }
  }
  const uint32_t aw = a.image.size.width;
  const uint32_t ah = a.image.size.height;
  const uint32_t bw = b.image.size.width;
  const uint32_t bh = b.image.size.height;
  const uint32_t merged_w = std::max(aw, bw);
  const uint32_t merged_h = std::max(ah, bh);
  AppendMaterializeDefaultFrameCrops(&a.frames, aw, ah, merged_w, merged_h);
  AppendMaterializeDefaultFrameCrops(&b.frames, bw, bh, merged_w, merged_h);

  const uint32_t ft_last = a.frames.back().frame.frame_type;
  if (ft_last == kFrameTypeLF || ft_last == kFrameTypeReferenceOnly) {
    *err =
        "primary codestream's last frame must be regular or skip-progressive";
    return false;
  }
  if (!b.frames.back().frame.is_last) {
    *err = "append codestream has no terminal is_last frame";
    return false;
  }
  const size_t cut_b = b.frames[0].original_frame_byte_offset;
  if (cut_b > append_cs.size()) {
    *err = "invalid first-frame offset in append codestream";
    return false;
  }
  const int64_t byte_delta =
      static_cast<int64_t>(primary_cs.size()) - static_cast<int64_t>(cut_b);
  if (byte_delta < 0) {
    *err = "invalid append layout (negative offset delta)";
    return false;
  }

  std::vector<uint8_t> combined;
  combined.reserve(primary_cs.size() + (append_cs.size() - cut_b));
  combined.insert(combined.end(), primary_cs.begin(), primary_cs.end());
  combined.insert(combined.end(), append_cs.begin() + cut_b,
                  append_cs.end());

  ParsedCodestream merged;
  merged.image = a.image;
  merged.image.size.width = merged_w;
  merged.image.size.height = merged_h;
  a.frames.back().frame.is_last = false;
  merged.frames = std::move(a.frames);
  merged.frames.reserve(merged.frames.size() + b.frames.size());
  for (size_t i = 0; i < b.frames.size(); ++i) {
    FramedUnit fu = b.frames[i];
    ShiftFramedUnitForAppend(&fu, byte_delta);
    merged.frames.push_back(std::move(fu));
  }
  // Standalone dummy is full-canvas replace (no source bits). After merge the
  // frame becomes a partial overlay on a larger canvas. kAdd with source 1
  // adds the zero patch to reference slot 1; that slot must hold the composited
  // image from the previous frame, so force the former primary tail to
  // save_as_reference=1 when it was is_last (otherwise nothing valid is in slot 1).
  if (IsBuiltinDummyTailCodestreamBytes(append_cs)) {
    if (merged.frames.size() < 2) {
      *err = "internal error: builtin dummy append without prior frame";
      return false;
    }
    FramedUnit& prev = merged.frames[merged.frames.size() - 2];
    FrameHeader* pfh = &prev.frame;
    if (pfh->frame_type == kFrameTypeRegular ||
        pfh->frame_type == kFrameTypeSkipProgressive) {
      pfh->save_as_reference = 1;
    }
    FrameHeader* fh = &merged.frames.back().frame;
    if (fh->frame_type == kFrameTypeRegular ||
        fh->frame_type == kFrameTypeSkipProgressive) {
      fh->blending_info.mode = 1;  // kAdd
      fh->blending_info.source = 1;
    }
  }
  if (!merged.frames.back().frame.is_last) {
    *err = "internal error: merged last frame is not is_last";
    return false;
  }
  if (!WriteCodestream(merged, combined.data(), merged_out)) {
    *err = "failed to rewrite merged codestream";
    return false;
  }
  return true;
}

}  // namespace jxltran
