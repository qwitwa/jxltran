// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Photon-noise LUT matching lib/jxl/enc_photon_noise.cc (ISO → 8×10-bit
// bitstream values) for jxltran.

#ifndef TOOLS_JXLTRAN_PHOTON_NOISE_H_
#define TOOLS_JXLTRAN_PHOTON_NOISE_H_

#include <array>
#include <cstddef>
#include <cstdint>

namespace jxltran {

static constexpr size_t kNoiseLutPoints = 8;
static constexpr float kNoisePrecision = 1024.0f;
static constexpr float kNoiseLutMax = 1023.4999f / kNoisePrecision;

// Same model as jxl::SimulatePhotonNoise (lib/jxl/enc_photon_noise.cc).
void SimulatePhotonNoise(size_t xsize, size_t ysize, float iso,
                         std::array<float, kNoiseLutPoints>* lut);

bool NoiseLutHasAny(const std::array<float, kNoiseLutPoints>& lut);

// Encode/decode the 8×10-bit noise block (lib/jxl/enc_noise.cc /
// lib/jxl/dec_noise.cc).
void EncodeNoiseLutBits(const std::array<float, kNoiseLutPoints>& lut,
                         std::array<uint8_t, 10>* out_bytes);
bool DecodeNoiseLutBits(const std::array<uint8_t, 10>& in_bytes,
                        std::array<float, kNoiseLutPoints>* lut);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_PHOTON_NOISE_H_
