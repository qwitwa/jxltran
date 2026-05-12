// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "restoration_filter.h"

#include <cmath>
#include <cstring>

#include "bits.h"

namespace jxltran {

float F16BitsToFloat(uint16_t h) {
  const uint32_t s = (h >> 15) & 0x1u;
  const uint32_t e = (h >> 10) & 0x1fu;
  const uint32_t m = h & 0x3ffu;
  if (e == 0) {
    if (m == 0) return s ? -0.f : 0.f;
    uint32_t mant = m;
    int exp = -14;
    while ((mant & 0x400) == 0) {
      mant <<= 1;
      --exp;
    }
    mant &= 0x3ffu;
    const uint32_t f =
        (s << 31) | static_cast<uint32_t>((exp + 127) << 23) | (mant << 13);
    float out;
    memcpy(&out, &f, 4);
    return out;
  }
  if (e == 31) {
    if (m == 0) {
      return s ? -INFINITY : INFINITY;
    }
    return NAN;
  }
  const uint32_t f = (s << 31) | ((e - 15 + 127) << 23) | (m << 13);
  float out;
  memcpy(&out, &f, 4);
  return out;
}

uint16_t FloatToF16Bits(float value) {
  uint32_t f;
  memcpy(&f, &value, 4);
  const uint32_t s = (f >> 16) & 0x8000u;
  uint32_t e = (f >> 23) & 0xffu;
  uint32_t m = f & 0x7fffffu;
  if (e == 255) {
    return static_cast<uint16_t>(s | 0x7c00u | (m ? 0x200u : 0u));
  }
  if (e == 0 && m == 0) return static_cast<uint16_t>(s);
  int32_t new_e = static_cast<int32_t>(e) - 127 + 15;
  if (new_e <= 0) {
    uint32_t mm = m | 0x800000u;
    int shift = 1 - new_e;
    if (shift > 24) return static_cast<uint16_t>(s);
    mm >>= static_cast<uint32_t>(shift);
    return static_cast<uint16_t>(s | (mm >> 13));
  }
  if (new_e >= 31) return static_cast<uint16_t>(s | 0x7c00u);
  return static_cast<uint16_t>(s | (static_cast<uint32_t>(new_e) << 10) |
                               (m >> 13));
}

static bool ReadExtensions(BitReader& br, ParsedRestorationFilter* out) {
  out->extensions_mask = ReadU64(br);
  out->extension_counts.clear();
  out->extension_payloads.clear();
  if (out->extensions_mask == 0) return true;
  uint64_t rem = out->extensions_mask;
  while (rem) {
    out->extension_counts.push_back(ReadU64(br));
    rem &= rem - 1;
  }
  size_t idx = 0;
  rem = out->extensions_mask;
  while (rem) {
    const uint64_t bc = out->extension_counts[idx++];
    std::vector<uint8_t> bytes((static_cast<size_t>(bc) + 7u) / 8u, 0);
    for (uint64_t i = 0; i < bc; ++i) {
      const uint32_t bit = br.ReadBits(1);
      bytes[static_cast<size_t>(i >> 3)] |=
          static_cast<uint8_t>(bit << (static_cast<int>(i) & 7));
    }
    out->extension_payloads.push_back(std::move(bytes));
    rem &= rem - 1;
  }
  return br.ok();
}

static void WriteExtensions(BitWriter& bw, const ParsedRestorationFilter& rf) {
  WriteU64(bw, rf.extensions_mask);
  if (rf.extensions_mask == 0) return;
  for (uint64_t c : rf.extension_counts) {
    WriteU64(bw, c);
  }
  size_t pi = 0;
  uint64_t rem = rf.extensions_mask;
  while (rem) {
    const uint64_t bc = rf.extension_counts[pi++];
    const std::vector<uint8_t>& bytes = rf.extension_payloads[pi - 1];
    for (uint64_t i = 0; i < bc; ++i) {
      const uint32_t bit =
          (bytes[static_cast<size_t>(i >> 3)] >> (static_cast<int>(i) & 7)) & 1;
      bw.WriteBits(1, bit);
    }
    rem &= rem - 1;
  }
}

static bool GabKernelOk(float w1, float w2) {
  return std::abs(1.0f + (w1 + w2) * 4.0f) >= 1e-8f;
}

bool ReadParsedRestorationFilter(BitReader& br, bool modular_frame,
                                 ParsedRestorationFilter* out) {
  ParsedRestorationFilter rf;
  rf.all_default = br.ReadBool();
  if (rf.all_default) {
    *out = rf;
    return br.ok();
  }

  rf.gab = br.ReadBool();
  if (rf.gab) {
    rf.gab_custom = br.ReadBool();
    if (rf.gab_custom) {
      rf.gab_x_weight1 = F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      rf.gab_x_weight2 = F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      if (!GabKernelOk(rf.gab_x_weight1, rf.gab_x_weight2)) return false;
      rf.gab_y_weight1 = F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      rf.gab_y_weight2 = F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      if (!GabKernelOk(rf.gab_y_weight1, rf.gab_y_weight2)) return false;
      rf.gab_b_weight1 = F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      rf.gab_b_weight2 = F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      if (!GabKernelOk(rf.gab_b_weight1, rf.gab_b_weight2)) return false;
    }
  }

  rf.epf_iters = br.ReadBits(2);
  if (rf.epf_iters > 0) {
    if (!modular_frame) {
      rf.epf_sharp_custom = br.ReadBool();
      if (rf.epf_sharp_custom) {
        for (int i = 0; i < kEpfSharpEntries; ++i) {
          rf.epf_sharp_lut[static_cast<size_t>(i)] =
              F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
        }
      }
    }

    rf.epf_weight_custom = br.ReadBool();
    if (rf.epf_weight_custom) {
      for (int i = 0; i < 3; ++i) {
        rf.epf_channel_scale[static_cast<size_t>(i)] =
            F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      }
      rf.epf_pass1_zeroflush =
          F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      rf.epf_pass2_zeroflush =
          F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
    }

    rf.epf_sigma_custom = br.ReadBool();
    if (rf.epf_sigma_custom) {
      if (!modular_frame) {
        rf.epf_quant_mul = F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      }
      rf.epf_pass0_sigma_scale =
          F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      rf.epf_pass2_sigma_scale =
          F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      rf.epf_border_sad_mul =
          F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
    }
    if (modular_frame) {
      rf.epf_sigma_for_modular =
          F16BitsToFloat(static_cast<uint16_t>(br.ReadF16()));
      if (rf.epf_sigma_for_modular < 1e-8f) return false;
    }
  }

  if (!ReadExtensions(br, &rf)) return false;
  *out = rf;
  return br.ok();
}

bool WriteParsedRestorationFilter(BitWriter& bw, bool modular_frame,
                                  const ParsedRestorationFilter& rf) {
  bw.WriteBool(rf.all_default);
  if (rf.all_default) return true;

  bw.WriteBool(rf.gab);
  if (rf.gab) {
    bw.WriteBool(rf.gab_custom);
    if (rf.gab_custom) {
      if (!GabKernelOk(rf.gab_x_weight1, rf.gab_x_weight2) ||
          !GabKernelOk(rf.gab_y_weight1, rf.gab_y_weight2) ||
          !GabKernelOk(rf.gab_b_weight1, rf.gab_b_weight2)) {
        return false;
      }
      bw.WriteF16(FloatToF16Bits(rf.gab_x_weight1));
      bw.WriteF16(FloatToF16Bits(rf.gab_x_weight2));
      bw.WriteF16(FloatToF16Bits(rf.gab_y_weight1));
      bw.WriteF16(FloatToF16Bits(rf.gab_y_weight2));
      bw.WriteF16(FloatToF16Bits(rf.gab_b_weight1));
      bw.WriteF16(FloatToF16Bits(rf.gab_b_weight2));
    }
  }

  bw.WriteBits(2, rf.epf_iters & 3u);
  if (rf.epf_iters > 0) {
    if (!modular_frame) {
      bw.WriteBool(rf.epf_sharp_custom);
      if (rf.epf_sharp_custom) {
        for (int i = 0; i < kEpfSharpEntries; ++i) {
          bw.WriteF16(FloatToF16Bits(rf.epf_sharp_lut[static_cast<size_t>(i)]));
        }
      }
    }

    bw.WriteBool(rf.epf_weight_custom);
    if (rf.epf_weight_custom) {
      for (int i = 0; i < 3; ++i) {
        bw.WriteF16(
            FloatToF16Bits(rf.epf_channel_scale[static_cast<size_t>(i)]));
      }
      bw.WriteF16(FloatToF16Bits(rf.epf_pass1_zeroflush));
      bw.WriteF16(FloatToF16Bits(rf.epf_pass2_zeroflush));
    }

    bw.WriteBool(rf.epf_sigma_custom);
    if (rf.epf_sigma_custom) {
      if (!modular_frame) {
        bw.WriteF16(FloatToF16Bits(rf.epf_quant_mul));
      }
      bw.WriteF16(FloatToF16Bits(rf.epf_pass0_sigma_scale));
      bw.WriteF16(FloatToF16Bits(rf.epf_pass2_sigma_scale));
      bw.WriteF16(FloatToF16Bits(rf.epf_border_sad_mul));
    }
    if (modular_frame) {
      if (rf.epf_sigma_for_modular < 1e-8f) return false;
      bw.WriteF16(FloatToF16Bits(rf.epf_sigma_for_modular));
    }
  }

  WriteExtensions(bw, rf);
  return true;
}

ParsedRestorationFilter DefaultRestorationFilter(bool modular_frame) {
  ParsedRestorationFilter rf;
  rf.all_default = false;
  rf.gab = true;
  rf.gab_custom = false;
  // lib/jxl/enc_frame.cc LoopFilterFromParams: modular defaults to no EPF;
  // VarDCT uses two EPF passes when loop filter is fully specified.
  rf.epf_iters = modular_frame ? 0u : 2u;
  rf.epf_sharp_custom = false;
  rf.epf_weight_custom = false;
  rf.epf_sigma_custom = false;
  if (modular_frame) {
    rf.epf_sigma_for_modular = 1.0f;
  }
  rf.extensions_mask = 0;
  return rf;
}

}  // namespace jxltran
