// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// JXL image header structs and parser/writer.
// Supports both want_icc==false and want_icc==true; for the latter the
// ICC profile entropy stream is parsed to find the byte boundary.
// See JXL spec Annexes A (Header syntax), B (Image header), C (Colour
// encoding), and parts of E (Colour transforms).

#ifndef TOOLS_JXLTRAN_IMAGE_HEADER_H_
#define TOOLS_JXLTRAN_IMAGE_HEADER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bits.h"

namespace jxltran {

// ── SizeHeader ───────────────────────────────────────────────────────────────

struct SizeHeader {
  uint32_t width = 1;
  uint32_t height = 1;
};

// ── PreviewHeader
// ─────────────────────────────────────────────────────────────

struct PreviewHeader {
  uint32_t width = 1;
  uint32_t height = 1;
};

// ── AnimationHeader
// ───────────────────────────────────────────────────────────

struct AnimationHeader {
  uint32_t tps_numerator = 0;
  uint32_t tps_denominator = 0;
  uint32_t num_loops = 0;
  bool have_timecodes = false;
};

// ── BitDepth
// ──────────────────────────────────────────────────────────────────

struct BitDepth {
  bool float_sample = false;
  uint32_t bits_per_sample = 8;
  uint32_t exp_bits = 0;  // only if float_sample
};

// ── ColourEncoding
// ────────────────────────────────────────────────────────────

enum class ColourSpace : uint32_t {
  kRGB = 0,
  kGrey = 1,
  kXYB = 2,
  kUnknown = 3
};

enum class WhitePoint : uint32_t { kD65 = 1, kCustom = 2, kE = 10, kDCI = 11 };

enum class Primaries : uint32_t { kSRGB = 1, kCustom = 2, k2100 = 9, kP3 = 11 };

enum class RenderingIntent : uint32_t {
  kPerceptual = 0,
  kRelative = 1,
  kSaturation = 2,
  kAbsolute = 3
};

// Customxy stores a CIE xy chromaticity coordinate pair as U32 raw values.
struct Customxy {
  uint32_t ux = 0;
  uint32_t uy = 0;
};

struct CustomTransferFunction {
  bool have_gamma = false;
  uint32_t gamma = 0;               // if have_gamma (u24, exponent/1e7)
  uint32_t transfer_function = 13;  // Enum(TransferFunction); kSRGB=13
};

struct ColourEncoding {
  bool all_default = true;
  // Fields below are only present in the bitstream if !all_default:
  bool want_icc = false;
  ColourSpace colour_space = ColourSpace::kRGB;
  WhitePoint white_point = WhitePoint::kD65;
  Customxy white;  // if white_point == kCustom
  Primaries primaries = Primaries::kSRGB;
  Customxy red, green, blue;  // if primaries == kCustom
  CustomTransferFunction tf;  // if use_desc (= !want_icc)
  RenderingIntent rendering_intent = RenderingIntent::kRelative;  // if use_desc
};

// ── ToneMapping
// ───────────────────────────────────────────────────────────────

struct ToneMapping {
  bool all_default = true;
  uint16_t intensity_target = 0;  // F16 raw bits; decoded = 255.0f
  uint16_t min_nits = 0;          // F16 raw bits; decoded = 0
  bool relative_to_max_display = false;
  uint16_t linear_below = 0;  // F16 raw bits; decoded = 0
};

// ── ExtraChannelInfo
// ──────────────────────────────────────────────────────────

enum class ExtraChannelType : uint32_t {
  kAlpha = 0,
  kDepth = 1,
  kSpotColour = 2,
  kSelectionMask = 3,
  kBlack = 4,
  kCFA = 5,
  kThermal = 6,
  kNonOptional = 15,
  kOptional = 16
};

struct ExtraChannelInfo {
  // In the spec, the "d_alpha" Bool() is the equivalent of all_default:
  // if true, this is a default alpha channel (no further fields read).
  bool d_alpha = true;
  // The following fields are only present when d_alpha == false:
  ExtraChannelType type = ExtraChannelType::kAlpha;
  BitDepth bit_depth;
  uint32_t dim_shift = 0;
  std::vector<uint8_t> name;      // UTF-8, length from name_len
  bool alpha_associated = false;  // if type == kAlpha (and !d_alpha)
  // kSpotColour:
  uint16_t spot_red = 0, spot_green = 0, spot_blue = 0, spot_solidity = 0;
  // kCFA:
  uint32_t cfa_channel = 1;
};

// ── Extensions
// ────────────────────────────────────────────────────────────────

// Extensions are stored as opaque bits; we don't interpret them.
struct Extensions {
  uint64_t mask = 0;
  // For each set bit in mask (in ascending bit-index order):
  std::vector<uint64_t> bit_counts;     // number of bits for that extension
  std::vector<std::vector<bool>> data;  // the raw bits
};

// ── OpsinInverseMatrix
// ────────────────────────────────────────────────────────

struct OpsinInverseMatrix {
  bool all_default = true;
  // F16 raw bits for the 3×3 inverse matrix (row-major), 3 opsin biases,
  // and 4 quant biases (matches libjxl OpsinInverseMatrix::VisitFields):
  uint16_t inv_mat[9] = {};
  uint16_t opsin_biases[3] = {};
  uint16_t quant_biases[4] = {};
};

// ── ImageMetadata
// ─────────────────────────────────────────────────────────────

struct ImageMetadata {
  bool all_default = true;

  // extra_fields section (present when !all_default):
  //   extra_fields is computed, not stored: true iff orientation != 1 ||
  //   have_intr_size || have_preview || have_animation.
  uint32_t orientation = 1;  // 1..8; Exif orientation semantics
  bool have_intr_size = false;
  SizeHeader intrinsic_size;
  bool have_preview = false;
  PreviewHeader preview;
  bool have_animation = false;
  AnimationHeader animation;

  // Fields present when !all_default:
  BitDepth bit_depth;
  bool modular_16bit_buffers = true;
  uint32_t num_extra = 0;
  std::vector<ExtraChannelInfo> ec_info;
  bool xyb_encoded = true;
  ColourEncoding colour_encoding;

  // Present when extra_fields (computed):
  ToneMapping tone_mapping;

  // Present when !all_default:
  Extensions extensions;

  // CustomTransformData (always present in the bitstream):
  bool default_m = true;
  // Present when !default_m && xyb_encoded:
  OpsinInverseMatrix opsin_inverse_matrix;
  // Present when !default_m:
  uint32_t cw_mask = 0;
  std::vector<uint16_t> up2_weight;  // 15 F16 values if bit 0 of cw_mask set
  std::vector<uint16_t> up4_weight;  // 55 F16 values if bit 1 of cw_mask set
  std::vector<uint16_t> up8_weight;  // 210 F16 values if bit 2 of cw_mask set
};

// ── ImageHeader
// ───────────────────────────────────────────────────────────────

struct ImageHeader {
  SizeHeader size;
  ImageMetadata metadata;

  // Populated by ReadImageHeader when metadata.colour_encoding.want_icc==true:
  // the range of bits in the original stream occupied by the ICC entropy-coded
  // block (not including the final byte-alignment padding).
  size_t icc_start_bit = 0;
  size_t icc_end_bit = 0;
  std::vector<uint8_t> icc_bytes;  // ICC.1 profile bytes when expansion succeeds
};

// ── API
// ───────────────────────────────────────────────────────────────────────

// Parses the JXL image header (including the 0xFF0A signature) from |br|.
// On success, |br| is advanced past the last bit of the header (including
// byte alignment after any embedded ICC profile entropy stream).
// Returns false if the signature is invalid or the bitstream is malformed.
bool ReadImageHeader(BitReader& br, ImageHeader* hdr);

// Writes the JXL image header (including the 0xFF0A signature) to |bw|.
// If hdr.metadata.colour_encoding.want_icc is true, the ICC entropy-coded
// bits are copied verbatim from |original_data| (the original codestream
// bytes), using the bit range [hdr.icc_start_bit, hdr.icc_end_bit).
void WriteImageHeader(BitWriter& bw, const ImageHeader& hdr,
                      const uint8_t* original_data = nullptr);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_IMAGE_HEADER_H_
