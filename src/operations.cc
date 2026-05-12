// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "operations.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <limits>

#include "codestream.h"
#include "frame_header.h"
#include "photon_noise.h"
#include "restoration_filter.h"

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

namespace {

constexpr float kGabDef1 = 1.1f * 0.104699568f;
constexpr float kGabDef2 = 1.1f * 0.055680538f;
constexpr float kGabIdentityEps = 1e-6f;
// Scale CLI |amount| into neighbor weights. Tuned so amount≈1.0 is a visible
// blur and a moderate sharpen on typical VarDCT files (identity baseline).
constexpr float kGabBlurAmountScale = 30.f; //2.35f;
constexpr float kGabSharpenAmountScale = 1.1f; //0.62f;

bool GabPairOk(float w1, float w2) {
  return std::abs(1.0f + (w1 + w2) * 4.0f) >= 1e-8f;
}

bool GabRfOk(const ParsedRestorationFilter& rf) {
  if (!rf.gab || !rf.gab_custom) return true;
  return GabPairOk(rf.gab_x_weight1, rf.gab_x_weight2) &&
         GabPairOk(rf.gab_y_weight1, rf.gab_y_weight2) &&
         GabPairOk(rf.gab_b_weight1, rf.gab_b_weight2);
}

// Effective "no Gaborish" in the decoder: gab off, or explicit all-zero
// neighbor weights (identity kernel: center 1, 1+4*(w1+w2)=1).
bool IdentityGabWeights(const ParsedRestorationFilter& rf) {
  if (!rf.gab) return true;
  if (!rf.gab_custom) return false;
  return std::abs(rf.gab_x_weight1) < kGabIdentityEps &&
         std::abs(rf.gab_x_weight2) < kGabIdentityEps &&
         std::abs(rf.gab_y_weight1) < kGabIdentityEps &&
         std::abs(rf.gab_y_weight2) < kGabIdentityEps &&
         std::abs(rf.gab_b_weight1) < kGabIdentityEps &&
         std::abs(rf.gab_b_weight2) < kGabIdentityEps;
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

}  // namespace

bool ApplyGabArgs(ParsedCodestream* cs, const GabArgs& args) {
  if (args.kind == GabArgs::Kind::kNone) return true;
  size_t applied = 0;
  for (FramedUnit& fu : cs->frames) {
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      continue;
    }
    FrameHeader* fh = &fu.frame;
    if (fh->restoration.all_default) {
      // Only the all-default flag was read; EPF fields were never parsed.
      // Expand to explicit libjxl defaults (LoopFilter::SetDefault), then keep
      // Gaborish off so IdentityGabWeights stays true until blur/sharpen
      // applies weights.
      const bool modular = (fh->encoding == kFrameEncModular);
      fh->restoration = DefaultRestorationFilter(modular);
      fh->restoration.gab = false;
      fh->restoration.gab_custom = false;
    }
    ParsedRestorationFilter* rf = &fh->restoration;
    const bool from_identity = IdentityGabWeights(*rf);
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
    } else if (args.kind == GabArgs::Kind::kBlur) {
      float w1[3];
      float w2[3];
      if (from_identity) {
        // libjxl's --gab-blur scales encoder defaults; with gab off that
        // baseline is identity (zero neighbor weights), not kGabDef*.
        const float s = args.amount * kGabBlurAmountScale;
        for (int c = 0; c < 3; ++c) {
          w1[c] = kGabDef1 * s;
          w2[c] = kGabDef2 * s;
        }
      } else {
        const float t = 1.f + args.amount * kGabBlurAmountScale;
        for (int c = 0; c < 3; ++c) {
          w1[c] = kGabDef1 * t;
          w2[c] = kGabDef2 * t;
        }
      }
      SetStandardGab(rf, w1, w2);
    } else if (args.kind == GabArgs::Kind::kSharpen) {
      float w1[3];
      float w2[3];
      if (from_identity) {
        // From zero weights: same delta as scaling down from kGabDef (sharpen
        // path).
        const float s = args.amount * kGabSharpenAmountScale;
        for (int c = 0; c < 3; ++c) {
          w1[c] = -kGabDef1 * s;
          w2[c] = -kGabDef2 * s;
        }
      } else {
        const float t = 1.f - args.amount * kGabSharpenAmountScale;
        for (int c = 0; c < 3; ++c) {
          w1[c] = kGabDef1 * t;
          w2[c] = kGabDef2 * t;
        }
      }
      SetStandardGab(rf, w1, w2);
    }
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
        const uint32_t lid = fu.toc_perm[q];
        if (lid >= fu.toc_decoded_sizes.size()) return false;
        off += fu.toc_decoded_sizes[lid];
      }
      *out_bytes = off;
      return true;
    }
  }
  return false;
}

}  // namespace

bool ApplyPhotonNoiseIso(ParsedCodestream* cs, bool change, float iso) {
  if (!change) return true;

  const uint32_t canvas_w = cs->image.size.width;
  const uint32_t canvas_h = cs->image.size.height;

  for (FramedUnit& fu : cs->frames) {
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

    if ((fh->flags & kFrameFlagPatches) != 0 ||
        (fh->flags & kFrameFlagSplines) != 0) {
      fprintf(stderr,
              "jxltran: --set-photon-noise-iso is not supported when patches or "
              "splines are enabled on a frame (bitstream layout is not "
              "handled).\n");
      return false;
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
    const size_t patch_abs = body0 + sec0_bytes * 8;
    const size_t body_bits_end = body0 + fu.body_bit_length;

    const bool had_noise = (fh->flags & kFrameFlagNoise) != 0;

    if ((patch_abs & 7u) != 0) {
      fprintf(stderr, "jxltran: photon noise: internal error (unaligned patch)\n");
      return false;
    }

    if (iso <= 0.f) {
      if (!had_noise) {
        continue;
      }
      if (patch_abs + 80 > body_bits_end) {
        fprintf(stderr,
                "jxltran: photon noise: noise LUT extends past frame body\n");
        return false;
      }
      if (!fu.toc_perm.empty()) {
        fprintf(stderr,
                "jxltran: photon noise: cannot change DC global size with TOC "
                "permutation present\n");
        return false;
      }
      fh->flags &= ~kFrameFlagNoise;
      fh->all_default = false;
      fu.photon_noise_edit = true;
      fu.photon_noise_delta_bytes = -10;
      fu.photon_noise_patch_abs_bit = patch_abs;
      fu.photon_noise_toc_reencode = true;
      if (!fu.toc_decoded_sizes.empty()) {
        const uint32_t s0 = fu.toc_decoded_sizes[0];
        if (s0 < 10u) {
          fprintf(stderr,
                  "jxltran: photon noise: DC global section too small to remove "
                  "noise\n");
          return false;
        }
        fu.toc_decoded_sizes[0] = s0 - 10u;
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
      if (patch_abs + 80 > body_bits_end) {
        fprintf(stderr,
                "jxltran: photon noise: noise LUT extends past frame body\n");
        return false;
      }
      if (!fu.toc_perm.empty()) {
        fprintf(stderr,
                "jxltran: photon noise: cannot change DC global size with TOC "
                "permutation present\n");
        return false;
      }
      fh->flags &= ~kFrameFlagNoise;
      fh->all_default = false;
      fu.photon_noise_edit = true;
      fu.photon_noise_delta_bytes = -10;
      fu.photon_noise_patch_abs_bit = patch_abs;
      fu.photon_noise_toc_reencode = true;
      if (!fu.toc_decoded_sizes.empty()) {
        const uint32_t s0 = fu.toc_decoded_sizes[0];
        if (s0 < 10u) {
          fprintf(stderr,
                  "jxltran: photon noise: DC global section too small to remove "
                  "noise\n");
          return false;
        }
        fu.toc_decoded_sizes[0] = s0 - 10u;
      }
      fu.body_bit_length -= 80;
      continue;
    }

    EncodeNoiseLutBits(lut, &fu.photon_noise_new_bytes);

    if (had_noise) {
      if (patch_abs + 80 > body_bits_end) {
        fprintf(stderr,
                "jxltran: photon noise: noise LUT extends past frame body\n");
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

    if (patch_abs > body_bits_end) {
      fprintf(stderr,
              "jxltran: photon noise: DC-global start past frame body\n");
      return false;
    }

    fh->flags |= kFrameFlagNoise;
    fh->all_default = false;
    fu.photon_noise_edit = true;
    fu.photon_noise_delta_bytes = 10;
    fu.photon_noise_patch_abs_bit = patch_abs;
    fu.photon_noise_toc_reencode = true;
    if (!fu.toc_decoded_sizes.empty()) {
      fu.toc_decoded_sizes[0] += 10u;
    }
    fu.body_bit_length += 80;
    if (!fu.toc_perm.empty()) {
      fprintf(stderr,
              "jxltran: photon noise: cannot change DC global size with TOC "
              "permutation present\n");
      return false;
    }
  }

  return true;
}

}  // namespace jxltran
