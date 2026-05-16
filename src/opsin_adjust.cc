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

void Mat3Mul(const float A[9], const float B[9], float C[9]) {
  for (int j = 0; j < 3; ++j) {
    for (int i = 0; i < 3; ++i) {
      C[j * 3 + i] = A[j * 3 + 0] * B[0 * 3 + i] + A[j * 3 + 1] * B[1 * 3 + i] +
                     A[j * 3 + 2] * B[2 * 3 + i];
    }
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
  if (std::abs(params.exposure_ev) < kEps && std::abs(params.temperature) < kEps &&
      std::abs(params.tint) < kEps && std::abs(params.hue) < kEps) {
    return true;
  }

  ImageMetadata* meta = &cs->image.metadata;
  if (!meta->xyb_encoded) {
    fprintf(stderr,
            "jxltran: opsin matrix adjustments require xyb_encoded=true "
            "(XYB / photographic mode)\n");
    return false;
  }

  const float t = params.temperature / 100.f;
  const float u = params.tint / 100.f;
  const float h = params.hue / 100.f;
  if (!(std::isfinite(t) && std::abs(t) <= 1.f) ||
      !(std::isfinite(u) && std::abs(u) <= 1.f) ||
      !(std::isfinite(h) && std::abs(h) <= 1.f) ||
      !std::isfinite(params.exposure_ev)) {
    fprintf(stderr,
            "jxltran: invalid opsin slider value (non-finite or out of range)\n");
    return false;
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

  if (params.undo_inverse) {
    // Forward transform is M' = 2^ev · R(h) · D_tint(u) · D_temp(t) · M.
    // Undo (exact inverse): M = inv(D_temp(t)) · inv(D_tint(u)) · inv(R(h)) · 2^{-ev} · M'.
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
      const float angle = h * (5.f * kPi / 180.f);
      const float ch = std::cos(angle);
      const float sh = std::sin(angle);
      const float inv_r[9] = {ch, 0.f, -sh, 0.f, 1.f, 0.f, sh, 0.f, ch};
      Mat3Mul(inv_r, W, tmp);
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
  } else {
    const float kt = 0.32f * t;
    const float d_temp[9] = {1.f + kt, 0.f, 0.f, 0.f, 1.f, 0.f, 0.f, 0.f, 1.f - kt};

    const float mag = 0.22f * u;
    const float d_tint[9] = {1.f - mag, 0.f, 0.f, 0.f, 1.f + 2.f * mag, 0.f,
                             0.f,       0.f, 1.f - mag};

    const float angle = h * (5.f * kPi / 180.f);
    const float ch = std::cos(angle);
    const float sh = std::sin(angle);
    const float r_hue[9] = {ch, 0.f, sh, 0.f, 1.f, 0.f, -sh, 0.f, ch};

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

  // If the result matches libjxl encoder defaults, use the compact bitstream form
  // (default_m=true, no OpsinInverseMatrix entropy) so round-trips match files
  // that never stored a custom matrix (see --check_reversible / reversible_undo).
  const bool was_implicit_default = !explicit_opsin;
  // Loose enough for float×F16 round-trip on inv_mat after exposure EV scales
  // cancel (inverse undo must collapse back to encoder-default signaling).
  constexpr float kDefaultTol = 0.02f;
  const bool no_custom_weights = meta->cw_mask == 0 && meta->up2_weight.empty() &&
                                  meta->up4_weight.empty() &&
                                  meta->up8_weight.empty();
  const float neg_bias = -kOpsinAbsorbanceBias;
  const float default_bias3[3] = {neg_bias, neg_bias, neg_bias};
  if (no_custom_weights && FloatVecNear(M, kDefaultInvMat, 9, kDefaultTol) &&
      FloatVecNear(bias_rgb, default_bias3, 3, kDefaultTol) &&
      FloatVecNear(quant_bias, kDefaultQuantBias, 4, kDefaultTol)) {
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

}  // namespace jxltran
