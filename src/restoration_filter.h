// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Parsed RestorationFilter (appendix G) for jxltran — read/write without libjxl.

#ifndef TOOLS_JXLTRAN_RESTORATION_FILTER_H_
#define TOOLS_JXLTRAN_RESTORATION_FILTER_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxltran {

class BitReader;
class BitWriter;

static constexpr int kEpfSharpEntries = 8;

// Mirrors lib/jxl/loop_filter.h serialized fields (nonserialized_is_modular
// is derived from frame encoding).
struct ParsedRestorationFilter {
  bool all_default = false;

  bool gab = false;
  bool gab_custom = false;
  float gab_x_weight1 = 0;
  float gab_x_weight2 = 0;
  float gab_y_weight1 = 0;
  float gab_y_weight2 = 0;
  float gab_b_weight1 = 0;
  float gab_b_weight2 = 0;

  uint32_t epf_iters = 0;

  bool epf_sharp_custom = false;
  std::array<float, kEpfSharpEntries> epf_sharp_lut = {};

  bool epf_weight_custom = false;
  std::array<float, 3> epf_channel_scale = {};
  float epf_pass1_zeroflush = 0;
  float epf_pass2_zeroflush = 0;

  bool epf_sigma_custom = false;
  float epf_quant_mul = 0;
  float epf_pass0_sigma_scale = 0;
  float epf_pass2_sigma_scale = 0;
  float epf_border_sad_mul = 0;

  float epf_sigma_for_modular = 0;

  uint64_t extensions_mask = 0;
  std::vector<uint64_t> extension_counts;
  std::vector<std::vector<uint8_t>> extension_payloads;
};

bool ReadParsedRestorationFilter(BitReader& br, bool modular_frame,
                                 ParsedRestorationFilter* out);

bool WriteParsedRestorationFilter(BitWriter& bw, bool modular_frame,
                                  const ParsedRestorationFilter& rf);

// lib/jxl/loop_filter.cc defaults when the bitstream uses restoration
// !all_default with implicit gab / EPF defaults (not a single all_default bit).
ParsedRestorationFilter DefaultRestorationFilter(bool modular_frame);

// IEEE binary16 <-> float32 (for F16 fields in the bitstream).
float F16BitsToFloat(uint16_t bits);
uint16_t FloatToF16Bits(float value);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_RESTORATION_FILTER_H_
