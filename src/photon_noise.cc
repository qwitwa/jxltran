// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "photon_noise.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace jxltran {

namespace {

// lib/jxl/cms/opsin_params.h
constexpr float kOpsinAbsorbanceBias1 = 0.0037930732552754493f;

// lib/jxl/enc_photon_noise.cc
constexpr float kPhotonsPerLxSPerUm2 = 11260;
constexpr float kEffectiveQuantumEfficiency = 0.20f;
constexpr float kPhotoResponseNonUniformity = 0.005f;
constexpr float kInputReferredReadNoise = 3.f;
constexpr float kSensorAreaUm2 = 36000.f * 24000.f;

template <typename T>
constexpr T Square(const T x) {
  return x * x;
}
template <typename T>
constexpr T Cube(const T x) {
  return x * x * x;
}

}  // namespace

void SimulatePhotonNoise(size_t xsize, size_t ysize, float iso,
                         std::array<float, kNoiseLutPoints>* lut) {
  const float kOpsinAbsorbanceBiasCbrt = std::cbrt(kOpsinAbsorbanceBias1);
  const float h_18 = 10.f / iso;
  const float pixel_area_um2 =
      kSensorAreaUm2 / static_cast<float>(xsize * ysize);
  const float electrons_per_pixel_18 =
      kEffectiveQuantumEfficiency * kPhotonsPerLxSPerUm2 * h_18 *
      pixel_area_um2;

  for (size_t i = 0; i < kNoiseLutPoints; ++i) {
    const float scaled_index = i / (kNoiseLutPoints - 2.f);
    const float y = 2 * scaled_index;
    const float linear = std::max(
        0.f, Cube(y - kOpsinAbsorbanceBiasCbrt) + kOpsinAbsorbanceBias1);
    const float electrons_per_pixel =
        electrons_per_pixel_18 * (linear / 0.18f);
    const float noise =
        std::sqrt(Square(kInputReferredReadNoise) + electrons_per_pixel +
                  Square(kPhotoResponseNonUniformity * electrons_per_pixel));
    const float linear_noise = noise * (0.18f / electrons_per_pixel_18);
    const float opsin_derivative =
        (1.f / 3) /
        Square(std::cbrt(linear - kOpsinAbsorbanceBias1));
    const float opsin_noise = linear_noise * opsin_derivative;
    (*lut)[i] = std::clamp(
        opsin_noise /
            (0.22f * std::sqrt(2.f) * 1.13f),
        0.f, kNoiseLutMax);
  }
}

bool NoiseLutHasAny(const std::array<float, kNoiseLutPoints>& lut) {
  for (float v : lut) {
    if (std::abs(v) > 1e-3f) return true;
  }
  return false;
}

void EncodeNoiseLutBits(const std::array<float, kNoiseLutPoints>& lut,
                         std::array<uint8_t, 10>* out_bytes) {
  out_bytes->fill(0);
  size_t bit_pos = 0;
  for (float v : lut) {
    const int q = static_cast<int>(std::lround(v * kNoisePrecision));
    for (int b = 0; b < 10; ++b, ++bit_pos) {
      if ((q >> b) & 1) {
        (*out_bytes)[bit_pos >> 3] |= static_cast<uint8_t>(1u << (bit_pos & 7));
      }
    }
  }
}

bool DecodeNoiseLutBits(const std::array<uint8_t, 10>& in_bytes,
                        std::array<float, kNoiseLutPoints>* lut) {
  size_t bit_pos = 0;
  for (float& out : *lut) {
    int q = 0;
    for (int b = 0; b < 10; ++b, ++bit_pos) {
      if ((in_bytes[bit_pos >> 3] >> (bit_pos & 7)) & 1) q |= (1 << b);
    }
    if (q < 0 || q >= (1 << 10)) return false;
    out = static_cast<float>(q) / kNoisePrecision;
  }
  return true;
}

}  // namespace jxltran
