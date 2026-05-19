// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "opsin_adjust.h"

#include <cmath>
#include <cstring>
#include <cstdio>

#include "image_header.h"
#include "restoration_filter.h"

namespace jxltran {

namespace {

// lib/jxl/cms/opsin_params.h — kDefaultInverseOpsinAbsorbanceMatrix (row j, col i).
constexpr float kDefaultInvMat[9] = {
    11.031566901960783f, -9.866943921568629f, -0.16462299647058826f,
    -3.254147380392157f, 4.418770392156863f,   -0.16462299647058826f,
    -3.6588512862745097f, 2.7129230470588235f, 1.9459282392156863f};

constexpr float kOpsinAbsorbanceBias = 0.0037930732552754493f;

constexpr float kDefaultQuantBias[4] = {
    1.0f - 0.05465007330715401f,
    1.0f - 0.07005449891748593f,
    1.0f - 0.049935103337343655f,
    0.145f,
};

constexpr float kPi = 3.14159265358979323846f;

// Rec. 709 display luma weights (unnormalized); axis for chroma-plane rotation.
constexpr float k709Lr = 0.2126f;
constexpr float k709Lg = 0.7152f;
constexpr float k709Lb = 0.0722f;
// Hue slider uses h ∈ [−1,1] (hue/100); map to ±120° so endpoints are not
// redundant with a full ±180° wrap.
constexpr float kHueSliderRadiansPerUnit = kPi * 2.0f / 3.0f;

void Mat3Mul(const float A[9], const float B[9], float C[9]) {
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      C[j * 3 + i] = A[j * 3 + 0] * B[0 * 3 + i] + A[j * 3 + 1] * B[1 * 3 + i] +
                     A[j * 3 + 2] * B[2 * 3 + i];
    }
  }
}

// Rodrigues rotation about Rec. 709 luma: R = I + sin(θ) K + (1−cos(θ)) K² with
// K v = k × v. Preserves linear Rec. 709 luma; closer to an HSL/HSV hue nudge
// than rotating in the R–B plane with G fixed.
void FillRec709LumaHueRotationMatrix(float radians, float R[9]) {
  const float invl =
      1.f / std::sqrt(k709Lr * k709Lr + k709Lg * k709Lg + k709Lb * k709Lb);
  const float kx = k709Lr * invl;
  const float ky = k709Lg * invl;
  const float kz = k709Lb * invl;
  const float K[9] = {
      0.f, -kz, ky, kz, 0.f, -kx, -ky, kx, 0.f,
  };
  float K2[9];
  Mat3Mul(K, K, K2);
  const float st = std::sin(radians);
  const float omct = 1.f - std::cos(radians);
  for (int i = 0; i < 9; ++i) {
    const float id = (i == 0 || i == 4 || i == 8) ? 1.f : 0.f;
    R[i] = id + st * K[i] + omct * K2[i];
  }
}

void Mat3Scale(float s, const float A[9], float C[9]) {
  for (int k = 0; k < 9; ++k) C[k] = s * A[k];
}

bool FloatFinite9(const float m[9]) {
  for (int k = 0; k < 9; ++k) {
    if (!std::isfinite(m[k])) return false;
  }
  return true;
}

bool FloatVecNear(const float* a, const float* b, int n, float tol) {
  for (int i = 0; i < n; ++i) {
    if (std::abs(a[i] - b[i]) > tol) return false;
  }
  return true;
}

}  // namespace

bool ApplyOpsinAdjust(ParsedCodestream* cs, const OpsinAdjustParams& params,
                      bool* did_change) {
  if (did_change) *did_change = false;
  constexpr float kEps = 1e-6f;
  const bool any_bias = params.opsin_bias_xyb[0].set || params.opsin_bias_xyb[1].set ||
                         params.opsin_bias_xyb[2].set;
  const bool any_quant =
      params.quant_bias_delta[0].set || params.quant_bias_delta[1].set ||
      params.quant_bias_delta[2].set || params.quant_bias_delta[3].set;
  const bool any_slider = std::abs(params.exposure_ev) >= kEps ||
                          std::abs(params.temperature) >= kEps ||
                          std::abs(params.tint) >= kEps || std::abs(params.hue) >= kEps;
  if (!any_slider && !any_bias && !any_quant) {
    return true;
  }

  ImageMetadata* meta = &cs->image.metadata;
  if (!meta->xyb_encoded) {
    fprintf(stderr,
            "jxltran: opsin matrix adjustments require xyb_encoded=true "
            "(XYB / photographic mode)\n");
    return false;
  }

  constexpr float kMaxOpsinBiasXY = 0.003f;
  constexpr float kMaxOpsinBiasB = 0.03f;
  constexpr float kMaxQuantBiasDelta = 20.f;
  constexpr float kRangeEps = 1e-6f;
  for (int kb = 0; kb < 3; ++kb) {
    if (!params.opsin_bias_xyb[kb].set) continue;
    const float bv = params.opsin_bias_xyb[kb].value;
    const float lim = (kb < 2) ? kMaxOpsinBiasXY : kMaxOpsinBiasB;
    if (!std::isfinite(bv) || bv < -lim - kRangeEps || bv > lim + kRangeEps) {
      fprintf(stderr,
              "jxltran: opsin bias value (channel %d) out of range [-%g, %g] "
              "or non-finite\n",
              kb, static_cast<double>(lim), static_cast<double>(lim));
      return false;
    }
  }
  for (int kq = 0; kq < 4; ++kq) {
    if (!params.quant_bias_delta[kq].set) continue;
    const float qv = params.quant_bias_delta[kq].value;
    if (!std::isfinite(qv) || qv < -kMaxQuantBiasDelta - kRangeEps ||
        qv > kMaxQuantBiasDelta + kRangeEps) {
      fprintf(stderr,
              "jxltran: quant bias value (index %d) out of range [-%g, %g] or "
              "non-finite\n",
              kq, static_cast<double>(kMaxQuantBiasDelta),
              static_cast<double>(kMaxQuantBiasDelta));
      return false;
    }
  }

  if (params.undo_inverse && !any_slider) {
    fprintf(stderr,
            "jxltran: --opsin-inverse requires at least one of --opsin-exposure, "
            "--opsin-temperature, --opsin-tint, --opsin-hue\n");
    return false;
  }

  const float t = params.temperature / 100.f;
  const float u = params.tint / 100.f;
  // Hue slider −100..+100 → h ∈ [−1,1] → ±90° rotation (see kHueSliderRadiansPerUnit).
  const float h = params.hue / 100.f;
  if (any_slider || params.undo_inverse) {
    if (!(std::isfinite(t) && std::abs(t) <= 1.f) ||
        !(std::isfinite(u) && std::abs(u) <= 1.f) ||
        !(std::isfinite(h) && std::abs(h) <= 1.f) ||
        !std::isfinite(params.exposure_ev)) {
      fprintf(stderr,
              "jxltran: invalid opsin slider value (non-finite or out of range)\n");
      return false;
    }
  }

  // Non-default CustomTransformData with opsin_inverse_matrix.all_default means
  // "use encoder defaults" (matrix + biases not present in the bitstream).
  const bool explicit_opsin =
      !meta->default_m && !meta->opsin_inverse_matrix.all_default;

  float bias_rgb[3];
  float quant_bias[4];
  if (explicit_opsin) {
    for (int k = 0; k < 3; ++k) {
      bias_rgb[k] = F16BitsToFloat(meta->opsin_inverse_matrix.opsin_biases[k]);
    }
    for (int k = 0; k < 4; ++k) {
      quant_bias[k] = F16BitsToFloat(meta->opsin_inverse_matrix.quant_biases[k]);
    }
  } else {
    const float neg_bias = -kOpsinAbsorbanceBias;
    bias_rgb[0] = neg_bias;
    bias_rgb[1] = neg_bias;
    bias_rgb[2] = neg_bias;
    for (int k = 0; k < 4; ++k) quant_bias[k] = kDefaultQuantBias[k];
  }

  float M[9];
  if (explicit_opsin) {
    for (int k = 0; k < 9; ++k) {
      M[k] = F16BitsToFloat(meta->opsin_inverse_matrix.inv_mat[k]);
    }
  } else {
    for (int k = 0; k < 9; ++k) M[k] = kDefaultInvMat[k];
  }
  if (!FloatFinite9(M)) {
    fprintf(stderr, "jxltran: existing OpsinInverseMatrix contains invalid values\n");
    return false;
  }

  if (params.undo_inverse && any_slider) {
    // Forward transform is M' = 2^ev · R(h) · D_tint(u) · D_temp(t) · M.
    // R(h) rotates linear RGB about Rec. 709 luma (HSL-like hue). Undo:
    // M = inv(D_temp(t)) · inv(D_tint(u)) · inv(R(h)) · 2^{-ev} · M'.
    // Negating t/u/h in the forward matrices is *not* the inverse (diagonal WB
    // factors are not involutions in that sense).
    float W[9];
    std::memcpy(W, M, sizeof(W));
    float tmp[9];
    const float ev = params.exposure_ev;
    if (std::abs(ev) > kEps) {
      Mat3Scale(std::pow(2.f, -ev), W, tmp);
      std::memcpy(W, tmp, sizeof(W));
    }
    if (std::abs(h) > kEps) {
      const float angle = h * kHueSliderRadiansPerUnit;
      float r_inv[9];
      FillRec709LumaHueRotationMatrix(-angle, r_inv);
      Mat3Mul(r_inv, W, tmp);
      std::memcpy(W, tmp, sizeof(W));
    }
    if (std::abs(u) > kEps) {
      const float mag = 0.22f * u;
      const float inv_d_tint[9] = {1.f / (1.f - mag), 0.f, 0.f, 0.f,
                                   1.f / (1.f + 2.f * mag), 0.f, 0.f, 0.f,
                                   1.f / (1.f - mag)};
      Mat3Mul(inv_d_tint, W, tmp);
      std::memcpy(W, tmp, sizeof(W));
    }
    if (std::abs(t) > kEps) {
      const float kt = 0.32f * t;
      if (std::abs(1.f + kt) < 1e-5f || std::abs(1.f - kt) < 1e-5f) {
        fprintf(stderr,
                "jxltran: opsin temperature magnitude too large for inverse WB step\n");
        return false;
      }
      const float inv_d_temp[9] = {1.f / (1.f + kt), 0.f, 0.f, 0.f, 1.f, 0.f,
                                   0.f,            0.f, 1.f / (1.f - kt)};
      Mat3Mul(inv_d_temp, W, tmp);
      std::memcpy(W, tmp, sizeof(W));
    }
    std::memcpy(M, W, sizeof(M));
    if (!FloatFinite9(M)) {
      fprintf(stderr, "jxltran: opsin inverse adjustment produced non-finite matrix\n");
      return false;
    }
  } else if (any_slider) {
    const float kt = 0.32f * t;
    const float d_temp[9] = {1.f + kt, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f - kt};

    const float mag = 0.22f * u;
    const float d_tint[9] = {1.f - mag, 0.f, 0.f, 0.f, 1.f + 2.f * mag, 0.f,
                             0.f,       0.f, 1.f - mag};

    const float angle = h * kHueSliderRadiansPerUnit;
    float r_hue[9];
    FillRec709LumaHueRotationMatrix(angle, r_hue);

    float A[9], B[9];
    Mat3Mul(d_temp, M, A);
    Mat3Mul(d_tint, A, B);
    Mat3Mul(r_hue, B, A);
    const float exp_scale = std::pow(2.f, params.exposure_ev);
    Mat3Scale(exp_scale, A, M);
    if (!FloatFinite9(M)) {
      fprintf(stderr, "jxltran: opsin adjustment produced non-finite matrix\n");
      return false;
    }
  }

  for (int kb = 0; kb < 3; ++kb) {
    if (!params.opsin_bias_xyb[kb].set) continue;
    bias_rgb[kb] += params.opsin_bias_xyb[kb].value;
    if (!std::isfinite(bias_rgb[kb])) {
      fprintf(stderr, "jxltran: opsin bias adjustment produced non-finite bias\n");
      return false;
    }
  }

  for (int kq = 0; kq < 4; ++kq) {
    if (!params.quant_bias_delta[kq].set) continue;
    quant_bias[kq] += params.quant_bias_delta[kq].value;
    if (!std::isfinite(quant_bias[kq])) {
      fprintf(stderr,
              "jxltran: quant bias adjustment produced non-finite quant bias\n");
      return false;
    }
  }

  // If the result matches libjxl encoder defaults, use the compact bitstream form
  // (default_m=true, no OpsinInverseMatrix entropy) so round-trips match files
  // that never stored a custom matrix (see --check_reversible / reversible_undo).
  const bool was_implicit_default = !explicit_opsin;
  // Loose enough for float×F16 round-trip on inv_mat after exposure EV scales
  // cancel (inverse undo must collapse back to encoder-default signaling).
  constexpr float kInvMatDefaultTol = 0.02f;
  const bool no_custom_weights = meta->cw_mask == 0 && meta->up2_weight.empty() &&
                                  meta->up4_weight.empty() &&
                                  meta->up8_weight.empty();
  const float neg_bias = -kOpsinAbsorbanceBias;
  if (no_custom_weights && FloatVecNear(M, kDefaultInvMat, 9, kInvMatDefaultTol) &&
      FloatToF16Bits(bias_rgb[0]) == FloatToF16Bits(neg_bias) &&
      FloatToF16Bits(bias_rgb[1]) == FloatToF16Bits(neg_bias) &&
      FloatToF16Bits(bias_rgb[2]) == FloatToF16Bits(neg_bias) &&
      FloatToF16Bits(quant_bias[0]) == FloatToF16Bits(kDefaultQuantBias[0]) &&
      FloatToF16Bits(quant_bias[1]) == FloatToF16Bits(kDefaultQuantBias[1]) &&
      FloatToF16Bits(quant_bias[2]) == FloatToF16Bits(kDefaultQuantBias[2]) &&
      FloatToF16Bits(quant_bias[3]) == FloatToF16Bits(kDefaultQuantBias[3])) {
    meta->default_m = true;
    meta->opsin_inverse_matrix = OpsinInverseMatrix{};
    if (did_change) *did_change = !was_implicit_default;
    return true;
  }

  meta->default_m = false;
  OpsinInverseMatrix* oim = &meta->opsin_inverse_matrix;
  oim->all_default = false;
  for (int k = 0; k < 9; ++k) {
    oim->inv_mat[k] = FloatToF16Bits(M[k]);
  }
  for (int k = 0; k < 3; ++k) {
    oim->opsin_biases[k] = FloatToF16Bits(bias_rgb[k]);
  }
  for (int k = 0; k < 4; ++k) {
    oim->quant_biases[k] = FloatToF16Bits(quant_bias[k]);
  }

  if (did_change) *did_change = true;
  return true;
}

void ComputeOpsinBiasUndoToImplicitDefault(const ImageMetadata& meta,
                                           const bool set[3], float undo_delta[3]) {
  for (int k = 0; k < 3; ++k) undo_delta[k] = 0.f;
  if (!meta.xyb_encoded) return;
  const float def_bias = -kOpsinAbsorbanceBias;
  float cur[3];
  const bool explicit_opsin =
      !meta.default_m && !meta.opsin_inverse_matrix.all_default;
  if (explicit_opsin) {
    for (int k = 0; k < 3; ++k) {
      cur[k] = F16BitsToFloat(meta.opsin_inverse_matrix.opsin_biases[k]);
    }
  } else {
    cur[0] = cur[1] = cur[2] = def_bias;
  }
  for (int k = 0; k < 3; ++k) {
    if (set[k]) undo_delta[k] = def_bias - cur[k];
  }
}

void ComputeQuantBiasUndoToImplicitDefault(const ImageMetadata& meta,
                                           const bool set[4], float undo_delta[4]) {
  for (int k = 0; k < 4; ++k) undo_delta[k] = 0.f;
  if (!meta.xyb_encoded) return;
  float cur[4];
  const bool explicit_opsin =
      !meta.default_m && !meta.opsin_inverse_matrix.all_default;
  if (explicit_opsin) {
    for (int k = 0; k < 4; ++k) {
      cur[k] = F16BitsToFloat(meta.opsin_inverse_matrix.quant_biases[k]);
    }
  } else {
    for (int k = 0; k < 4; ++k) cur[k] = kDefaultQuantBias[k];
  }
  for (int k = 0; k < 4; ++k) {
    if (set[k]) undo_delta[k] = kDefaultQuantBias[k] - cur[k];
  }
}

}  // namespace jxltran
