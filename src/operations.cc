// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "operations.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>

#include "codestream.h"
#include "entropy.h"
#include "frame_header.h"
#include "photon_noise.h"
#include "printf_macros.h"
#include "lfglobal.h"
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

}  // namespace jxltran
