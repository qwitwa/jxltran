// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "image_header.h"

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "printf_macros.h"
#include "entropy.h"
#include "trace.h"

namespace jxltran {

// ── U64 ──────────────────────────────────────────────────────────────────────

uint64_t ReadU64(BitReader& br) {
  const uint32_t sel = br.ReadBits(2);
  if (sel == 0) return 0;
  if (sel == 1) return 1 + br.ReadBits(4);
  if (sel == 2) return 17 + br.ReadBits(8);
  uint64_t value = br.ReadBits(12);
  int shift = 12;
  while (br.ReadBits(1)) {
    if (shift == 60) {
      value += static_cast<uint64_t>(br.ReadBits(4)) << shift;
      break;
    }
    value += static_cast<uint64_t>(br.ReadBits(8)) << shift;
    shift += 8;
  }
  return value;
}

void WriteU64(BitWriter& bw, uint64_t value) {
  if (value == 0) {
    bw.WriteBits(2, 0);
    return;
  }
  if (value <= 16) {
    bw.WriteBits(2, 1);
    bw.WriteBits(4, value - 1);
    return;
  }
  if (value <= 272) {
    bw.WriteBits(2, 2);
    bw.WriteBits(8, value - 17);
    return;
  }
  bw.WriteBits(2, 3);
  bw.WriteBits(12, value & 0xFFF);
  value >>= 12;
  int shift = 12;
  while (true) {
    if (shift == 60) {
      bw.WriteBits(1, 1);
      bw.WriteBits(4, value & 0xF);
      return;
    }
    if (value == 0) {
      bw.WriteBits(1, 0);
      return;
    }
    bw.WriteBits(1, 1);
    bw.WriteBits(8, value & 0xFF);
    value >>= 8;
    shift += 8;
  }
}

// ── SizeHeader (matches lib/jxl/headers.cc SizeHeader::VisitFields / Set)
// ────────────────────────────────────────────────────────────────

namespace {

struct Rational {
  uint32_t num;
  uint32_t den;
  constexpr Rational(uint32_t n, uint32_t d) : num(n), den(d) {}
  uint32_t MulTruncate(uint32_t multiplicand) const {
    return static_cast<uint64_t>(multiplicand) * num / den;
  }
};

Rational FixedAspectRatios(uint32_t ratio) {
  constexpr Rational kRatios[7] = {
      Rational(1, 1),    Rational(12, 10), Rational(4, 3), Rational(3, 2),
      Rational(16, 9), Rational(5, 4),   Rational(2, 1)};
  return kRatios[ratio - 1];
}

uint32_t FindAspectRatio(uint32_t xsize, uint32_t ysize) {
  for (uint32_t r = 1; r < 8; ++r) {
    if (xsize == FixedAspectRatios(r).MulTruncate(ysize)) {
      return r;
    }
  }
  return 0;
}

constexpr uint32_t kBlockDim = 8;

bool ReadSizeHeader(BitReader& br, SizeHeader* s) {
  const size_t p0 = br.pos();
  const bool small = br.ReadBool();
  uint32_t ysize;
  if (small) {
    ysize = (1 + br.ReadBits(5)) * 8;
  } else {
    ysize =
        ReadU32(br, U32Dist::BitsOffset(9, 1), U32Dist::BitsOffset(13, 1),
                U32Dist::BitsOffset(18, 1), U32Dist::BitsOffset(30, 1));
  }
  const uint32_t ratio = br.ReadBits(3);
  uint32_t xsize;
  if (ratio != 0) {
    xsize = FixedAspectRatios(ratio).MulTruncate(ysize);
  } else if (small) {
    xsize = (1 + br.ReadBits(5)) * 8;
  } else {
    xsize =
        ReadU32(br, U32Dist::BitsOffset(9, 1), U32Dist::BitsOffset(13, 1),
                U32Dist::BitsOffset(18, 1), U32Dist::BitsOffset(30, 1));
  }
  s->width = xsize;
  s->height = ysize;
  TraceReadField(p0, "read.image.size",
                 "small=%s ratio=%" PRIu32 " width=%" PRIu32 " height=%" PRIu32
                 "",
                 small ? "true" : "false", ratio, s->width, s->height);
  return s->width > 0 && s->height > 0 && br.ok();
}

void WriteSizeHeader(BitWriter& bw, const SizeHeader& s) {
  const size_t p0 = bw.bit_pos();
  const uint32_t xsize = s.width;
  const uint32_t ysize = s.height;
  const uint32_t ratio = FindAspectRatio(xsize, ysize);
  const bool small =
      ysize <= 256 && (ysize % kBlockDim) == 0 &&
      (ratio != 0 || (xsize <= 256 && (xsize % kBlockDim) == 0));

  bw.WriteBool(small);
  if (small) {
    bw.WriteBits(5, ysize / 8 - 1);
  } else {
    WriteU32(bw, ysize, U32Dist::BitsOffset(9, 1), U32Dist::BitsOffset(13, 1),
             U32Dist::BitsOffset(18, 1), U32Dist::BitsOffset(30, 1));
  }
  bw.WriteBits(3, ratio);
  if (ratio == 0 && small) {
    bw.WriteBits(5, xsize / 8 - 1);
  }
  if (ratio == 0 && !small) {
    WriteU32(bw, xsize, U32Dist::BitsOffset(9, 1), U32Dist::BitsOffset(13, 1),
             U32Dist::BitsOffset(18, 1), U32Dist::BitsOffset(30, 1));
  }
  TraceWriteField(
      p0, "write.image.size",
      "small=%s ratio=%" PRIu32 " width=%" PRIu32 " height=%" PRIu32 "",
      small ? "true" : "false", ratio, s.width, s.height);
}

// ── PreviewHeader
// ─────────────────────────────────────────────────────────────

bool ReadPreviewHeader(BitReader& br, PreviewHeader* p) {
  const bool div8 = br.ReadBool();
  if (div8) {
    p->height = ReadU32(br, U32Dist::Imm(16), U32Dist::Imm(32),
                        U32Dist::BitsOffset(5, 1), U32Dist::BitsOffset(9, 33));
  } else {
    p->height =
        ReadU32(br, U32Dist::BitsOffset(6, 1), U32Dist::BitsOffset(8, 65),
                U32Dist::BitsOffset(10, 321), U32Dist::BitsOffset(12, 1345));
  }
  const uint32_t ratio = br.ReadBits(3);
  if (ratio != 0) {
    p->width = FixedAspectRatios(ratio).MulTruncate(p->height);
  } else if (div8) {
    p->width = ReadU32(br, U32Dist::Imm(16), U32Dist::Imm(32),
                       U32Dist::BitsOffset(5, 1), U32Dist::BitsOffset(9, 33));
  } else {
    p->width =
        ReadU32(br, U32Dist::BitsOffset(6, 1), U32Dist::BitsOffset(8, 65),
                U32Dist::BitsOffset(10, 321), U32Dist::BitsOffset(12, 1345));
  }
  return true;
}

void WritePreviewHeader(BitWriter& bw, const PreviewHeader& p) {
  bw.WriteBool(false);  // !div8
  WriteU32(bw, p.height, U32Dist::BitsOffset(6, 1), U32Dist::BitsOffset(8, 65),
           U32Dist::BitsOffset(10, 321), U32Dist::BitsOffset(12, 1345));
  bw.WriteBits(3, 0);  // ratio = 0
  WriteU32(bw, p.width, U32Dist::BitsOffset(6, 1), U32Dist::BitsOffset(8, 65),
           U32Dist::BitsOffset(10, 321), U32Dist::BitsOffset(12, 1345));
}

}  // namespace

// ── AnimationHeader
// ───────────────────────────────────────────────────────────

static bool ReadAnimationHeader(BitReader& br, AnimationHeader* a) {
  a->tps_numerator =
      ReadU32(br, U32Dist::Imm(100), U32Dist::Imm(1000),
              U32Dist::BitsOffset(10, 1), U32Dist::BitsOffset(30, 1));
  a->tps_denominator =
      ReadU32(br, U32Dist::Imm(1), U32Dist::Imm(1001),
              U32Dist::BitsOffset(8, 1), U32Dist::BitsOffset(10, 1));
  a->num_loops = ReadU32(br, U32Dist::Imm(0), U32Dist::Bits(3),
                         U32Dist::Bits(16), U32Dist::Bits(32));
  a->have_timecodes = br.ReadBool();
  return true;
}

static void WriteAnimationHeader(BitWriter& bw, const AnimationHeader& a) {
  WriteU32(bw, a.tps_numerator, U32Dist::Imm(100), U32Dist::Imm(1000),
           U32Dist::BitsOffset(10, 1), U32Dist::BitsOffset(30, 1));
  WriteU32(bw, a.tps_denominator, U32Dist::Imm(1), U32Dist::Imm(1001),
           U32Dist::BitsOffset(8, 1), U32Dist::BitsOffset(10, 1));
  WriteU32(bw, a.num_loops, U32Dist::Imm(0), U32Dist::Bits(3),
           U32Dist::Bits(16), U32Dist::Bits(32));
  bw.WriteBool(a.have_timecodes);
}

// ── BitDepth
// ──────────────────────────────────────────────────────────────────

static bool ReadBitDepth(BitReader& br, BitDepth* bd) {
  bd->float_sample = br.ReadBool();
  if (!bd->float_sample) {
    bd->bits_per_sample = ReadU32(br, U32Dist::Imm(8), U32Dist::Imm(10),
                                  U32Dist::Imm(12), U32Dist::BitsOffset(6, 1));
  } else {
    bd->bits_per_sample = ReadU32(br, U32Dist::Imm(32), U32Dist::Imm(16),
                                  U32Dist::Imm(24), U32Dist::BitsOffset(6, 1));
    bd->exp_bits = 1 + br.ReadBits(4);
  }
  return true;
}

static void WriteBitDepth(BitWriter& bw, const BitDepth& bd) {
  bw.WriteBool(bd.float_sample);
  if (!bd.float_sample) {
    WriteU32(bw, bd.bits_per_sample, U32Dist::Imm(8), U32Dist::Imm(10),
             U32Dist::Imm(12), U32Dist::BitsOffset(6, 1));
  } else {
    WriteU32(bw, bd.bits_per_sample, U32Dist::Imm(32), U32Dist::Imm(16),
             U32Dist::Imm(24), U32Dist::BitsOffset(6, 1));
    bw.WriteBits(4, bd.exp_bits - 1);
  }
}

// ── Customxy
// ──────────────────────────────────────────────────────────────────

static void ReadCustomxy(BitReader& br, Customxy* xy) {
  xy->ux = ReadU32(br, U32Dist::Bits(19), U32Dist::BitsOffset(19, 524288),
                   U32Dist::BitsOffset(20, 1048576),
                   U32Dist::BitsOffset(21, 2097152));
  xy->uy = ReadU32(br, U32Dist::Bits(19), U32Dist::BitsOffset(19, 524288),
                   U32Dist::BitsOffset(20, 1048576),
                   U32Dist::BitsOffset(21, 2097152));
}

static void WriteCustomxy(BitWriter& bw, const Customxy& xy) {
  WriteU32(bw, xy.ux, U32Dist::Bits(19), U32Dist::BitsOffset(19, 524288),
           U32Dist::BitsOffset(20, 1048576), U32Dist::BitsOffset(21, 2097152));
  WriteU32(bw, xy.uy, U32Dist::Bits(19), U32Dist::BitsOffset(19, 524288),
           U32Dist::BitsOffset(20, 1048576), U32Dist::BitsOffset(21, 2097152));
}

// ── ColourEncoding
// ────────────────────────────────────────────────────────────

static bool ReadColourEncoding(BitReader& br, ColourEncoding* ce) {
  ce->all_default = br.ReadBool();
  if (ce->all_default) return true;

  ce->want_icc = br.ReadBool();
  ce->colour_space = static_cast<ColourSpace>(ReadEnum(br));

  const bool use_desc = !ce->want_icc;
  const bool not_xyb = ce->colour_space != ColourSpace::kXYB;
  const bool has_primaries =
      use_desc && not_xyb && ce->colour_space != ColourSpace::kGrey;

  if (use_desc && not_xyb) {
    ce->white_point = static_cast<WhitePoint>(ReadEnum(br));
    if (ce->white_point == WhitePoint::kCustom) ReadCustomxy(br, &ce->white);
  }
  if (has_primaries) {
    ce->primaries = static_cast<Primaries>(ReadEnum(br));
    if (ce->primaries == Primaries::kCustom) {
      ReadCustomxy(br, &ce->red);
      ReadCustomxy(br, &ce->green);
      ReadCustomxy(br, &ce->blue);
    }
  }
  if (use_desc) {
    // CustomTransferFunction: implicit (no bits) for XYB, see
    // CustomTransferFunction::SetImplicit in
    // lib/jxl/color_encoding_internal.cc.
    const bool implicit_tf = (ce->colour_space == ColourSpace::kXYB);
    if (!implicit_tf) {
      ce->tf.have_gamma = br.ReadBool();
      if (ce->tf.have_gamma) {
        ce->tf.gamma = br.ReadBits(24);
      } else {
        ce->tf.transfer_function = ReadEnum(br);
      }
    }
    ce->rendering_intent = static_cast<RenderingIntent>(ReadEnum(br));
  }
  return true;
}

static void WriteColourEncoding(BitWriter& bw, const ColourEncoding& ce) {
  bw.WriteBool(ce.all_default);
  if (ce.all_default) return;

  bw.WriteBool(ce.want_icc);
  WriteEnum(bw, static_cast<uint32_t>(ce.colour_space));

  const bool use_desc = !ce.want_icc;
  const bool not_xyb = ce.colour_space != ColourSpace::kXYB;
  const bool has_primaries =
      use_desc && not_xyb && ce.colour_space != ColourSpace::kGrey;

  if (use_desc && not_xyb) {
    WriteEnum(bw, static_cast<uint32_t>(ce.white_point));
    if (ce.white_point == WhitePoint::kCustom) WriteCustomxy(bw, ce.white);
  }
  if (has_primaries) {
    WriteEnum(bw, static_cast<uint32_t>(ce.primaries));
    if (ce.primaries == Primaries::kCustom) {
      WriteCustomxy(bw, ce.red);
      WriteCustomxy(bw, ce.green);
      WriteCustomxy(bw, ce.blue);
    }
  }
  if (use_desc) {
    const bool implicit_tf = (ce.colour_space == ColourSpace::kXYB);
    if (!implicit_tf) {
      bw.WriteBool(ce.tf.have_gamma);
      if (ce.tf.have_gamma) {
        bw.WriteBits(24, ce.tf.gamma);
      } else {
        WriteEnum(bw, ce.tf.transfer_function);
      }
    }
    WriteEnum(bw, static_cast<uint32_t>(ce.rendering_intent));
  }
}

// ── ToneMapping
// ───────────────────────────────────────────────────────────────

static void ReadToneMapping(BitReader& br, ToneMapping* tm) {
  tm->all_default = br.ReadBool();
  if (tm->all_default) return;
  tm->intensity_target = br.ReadF16();
  tm->min_nits = br.ReadF16();
  tm->relative_to_max_display = br.ReadBool();
  tm->linear_below = br.ReadF16();
}

static void WriteToneMapping(BitWriter& bw, const ToneMapping& tm) {
  bw.WriteBool(tm.all_default);
  if (tm.all_default) return;
  bw.WriteF16(tm.intensity_target);
  bw.WriteF16(tm.min_nits);
  bw.WriteBool(tm.relative_to_max_display);
  bw.WriteF16(tm.linear_below);
}

// ── Extensions
// ────────────────────────────────────────────────────────────────

static void ReadExtensions(BitReader& br, Extensions* ext) {
  ext->mask = ReadU64(br);
  if (ext->mask == 0) return;
  // Count set bits, read bit counts for each.
  uint64_t remaining = ext->mask;
  while (remaining) {
    ext->bit_counts.push_back(ReadU64(br));
    remaining &= remaining - 1;
  }
  // Read the raw extension bits.
  for (uint64_t bc : ext->bit_counts) {
    std::vector<bool> bits;
    bits.reserve(static_cast<size_t>(bc));
    for (uint64_t i = 0; i < bc; ++i) {
      bits.push_back(br.ReadBits(1) != 0);
    }
    ext->data.push_back(std::move(bits));
  }
}

static void WriteExtensions(BitWriter& bw, const Extensions& ext) {
  WriteU64(bw, ext.mask);
  if (ext.mask == 0) return;
  for (uint64_t bc : ext.bit_counts) WriteU64(bw, bc);
  for (const auto& bits : ext.data) {
    for (bool b : bits) bw.WriteBool(b);
  }
}

// ── ExtraChannelInfo
// ──────────────────────────────────────────────────────────

static bool ReadExtraChannelInfo(BitReader& br, ExtraChannelInfo* ec) {
  ec->d_alpha = br.ReadBool();
  // name is always read, but name_len defaults to 0 when d_alpha == true.
  if (!ec->d_alpha) {
    ec->type = static_cast<ExtraChannelType>(ReadEnum(br));
    if (!ReadBitDepth(br, &ec->bit_depth)) return false;
    ec->dim_shift = ReadU32(br, U32Dist::Imm(0), U32Dist::Imm(3),
                            U32Dist::Imm(4), U32Dist::BitsOffset(3, 1));
    const uint32_t name_len =
        ReadU32(br, U32Dist::Imm(0), U32Dist::Bits(4),
                U32Dist::BitsOffset(5, 16), U32Dist::BitsOffset(10, 48));
    ec->name.resize(name_len);
    for (uint32_t i = 0; i < name_len; ++i) {
      ec->name[i] = static_cast<uint8_t>(br.ReadBits(8));
    }
    if (ec->type == ExtraChannelType::kAlpha) {
      ec->alpha_associated = br.ReadBool();
    }
  }
  if (ec->type == ExtraChannelType::kSpotColour) {
    ec->spot_red = br.ReadF16();
    ec->spot_green = br.ReadF16();
    ec->spot_blue = br.ReadF16();
    ec->spot_solidity = br.ReadF16();
  }
  if (ec->type == ExtraChannelType::kCFA) {
    ec->cfa_channel =
        ReadU32(br, U32Dist::Imm(1), U32Dist::Bits(2),
                U32Dist::BitsOffset(4, 3), U32Dist::BitsOffset(8, 19));
  }
  return true;
}

static void WriteExtraChannelInfo(BitWriter& bw, const ExtraChannelInfo& ec) {
  bw.WriteBool(ec.d_alpha);
  if (!ec.d_alpha) {
    WriteEnum(bw, static_cast<uint32_t>(ec.type));
    WriteBitDepth(bw, ec.bit_depth);
    WriteU32(bw, ec.dim_shift, U32Dist::Imm(0), U32Dist::Imm(3),
             U32Dist::Imm(4), U32Dist::BitsOffset(3, 1));
    WriteU32(bw, static_cast<uint32_t>(ec.name.size()), U32Dist::Imm(0),
             U32Dist::Bits(4), U32Dist::BitsOffset(5, 16),
             U32Dist::BitsOffset(10, 48));
    for (uint8_t c : ec.name) bw.WriteBits(8, c);
    if (ec.type == ExtraChannelType::kAlpha) {
      bw.WriteBool(ec.alpha_associated);
    }
  }
  if (ec.type == ExtraChannelType::kSpotColour) {
    bw.WriteF16(ec.spot_red);
    bw.WriteF16(ec.spot_green);
    bw.WriteF16(ec.spot_blue);
    bw.WriteF16(ec.spot_solidity);
  }
  if (ec.type == ExtraChannelType::kCFA) {
    WriteU32(bw, ec.cfa_channel, U32Dist::Imm(1), U32Dist::Bits(2),
             U32Dist::BitsOffset(4, 3), U32Dist::BitsOffset(8, 19));
  }
}

// ── OpsinInverseMatrix
// ────────────────────────────────────────────────────────

static void ReadOpsinInverseMatrix(BitReader& br, OpsinInverseMatrix* oim) {
  oim->all_default = br.ReadBool();
  if (oim->all_default) return;
  for (int i = 0; i < 9; ++i) oim->inv_mat[i] = br.ReadF16();
  for (int i = 0; i < 3; ++i) oim->opsin_biases[i] = br.ReadF16();
  for (int i = 0; i < 4; ++i) oim->quant_biases[i] = br.ReadF16();
}

static void WriteOpsinInverseMatrix(BitWriter& bw,
                                    const OpsinInverseMatrix& oim) {
  bw.WriteBool(oim.all_default);
  if (oim.all_default) return;
  for (int i = 0; i < 9; ++i) bw.WriteF16(oim.inv_mat[i]);
  for (int i = 0; i < 3; ++i) bw.WriteF16(oim.opsin_biases[i]);
  for (int i = 0; i < 4; ++i) bw.WriteF16(oim.quant_biases[i]);
}

// ── ImageMetadata
// ─────────────────────────────────────────────────────────────

static bool ReadImageMetadata(BitReader& br, ImageHeader* hdr) {
  ImageMetadata* m = &hdr->metadata;
  {
    const size_t p = br.pos();
    m->all_default = br.ReadBool();
    TraceReadField(p, "read.image.metadata.all_default", "%s",
                   m->all_default ? "true" : "false");
  }
  if (!m->all_default) {
    const size_t p = br.pos();
    const bool extra_fields = br.ReadBool();
    TraceReadField(p, "read.image.metadata.extra_fields", "%s",
                   extra_fields ? "true" : "false");
    if (extra_fields) {
      const size_t pori = br.pos();
      m->orientation = 1 + br.ReadBits(3);
      TraceReadField(pori, "read.image.metadata.orientation", "%u",
                     m->orientation);
      m->have_intr_size = br.ReadBool();
      if (m->have_intr_size) {
        if (!ReadSizeHeader(br, &m->intrinsic_size)) return false;
      }
      m->have_preview = br.ReadBool();
      if (m->have_preview) {
        if (!ReadPreviewHeader(br, &m->preview)) return false;
      }
      m->have_animation = br.ReadBool();
      if (m->have_animation) {
        if (!ReadAnimationHeader(br, &m->animation)) return false;
      }
    }
    if (!ReadBitDepth(br, &m->bit_depth)) return false;
    m->modular_16bit_buffers = br.ReadBool();
    m->num_extra =
        ReadU32(br, U32Dist::Imm(0), U32Dist::Imm(1), U32Dist::BitsOffset(4, 2),
                U32Dist::BitsOffset(12, 1));
    m->ec_info.resize(m->num_extra);
    for (uint32_t i = 0; i < m->num_extra; ++i) {
      if (!ReadExtraChannelInfo(br, &m->ec_info[i])) return false;
    }
    m->xyb_encoded = br.ReadBool();
    if (!ReadColourEncoding(br, &m->colour_encoding)) return false;
    if (extra_fields) {
      ReadToneMapping(br, &m->tone_mapping);
    }
    ReadExtensions(br, &m->extensions);
  }

  // CustomTransformData — always read.
  m->default_m = br.ReadBool();
  if (!m->default_m) {
    if (m->xyb_encoded) {
      ReadOpsinInverseMatrix(br, &m->opsin_inverse_matrix);
    }
    m->cw_mask = br.ReadBits(3);
    // Up2: 15 weights if bit 0 set
    if (m->cw_mask & 1) {
      m->up2_weight.resize(15);
      for (int i = 0; i < 15; ++i) m->up2_weight[i] = br.ReadF16();
    }
    // Up4: 55 weights if bit 1 set
    if (m->cw_mask & 2) {
      m->up4_weight.resize(55);
      for (int i = 0; i < 55; ++i) m->up4_weight[i] = br.ReadF16();
    }
    // Up8: 210 weights if bit 2 set
    if (m->cw_mask & 4) {
      m->up8_weight.resize(210);
      for (int i = 0; i < 210; ++i) m->up8_weight[i] = br.ReadF16();
    }
  }

  // ICC entropy stream follows transform on the wire (lib/jxl/encode.cc,
  // JxlDecoderReadAllHeaders in decode.cc), not inside ImageMetadata.
  if (m->colour_encoding.want_icc) {
    hdr->icc_start_bit = br.pos();
    TraceReadField(hdr->icc_start_bit, "read.image.icc_entropy", "(start)");
    if (!SkipICCStream(br, &hdr->icc_bytes)) return false;
    hdr->icc_end_bit = br.pos();
    TraceReadField(hdr->icc_end_bit, "read.image.icc_entropy_end",
                   "span_bits=%" PRIuS " decoded_bytes=%" PRIuS "",
                   hdr->icc_end_bit - hdr->icc_start_bit, hdr->icc_bytes.size());
  }
  return br.ok();
}

static void WriteImageMetadata(BitWriter& bw, const ImageMetadata& m) {
  // Match lib/jxl/image_metadata.cc ImageMetadata::VisitFields: extra_fields
  // is also true when tone mapping is non-default.
  const bool extra_fields = (m.orientation != 1) || m.have_intr_size ||
                            m.have_preview || m.have_animation ||
                            !m.tone_mapping.all_default;

  bw.WriteBool(m.all_default);
  if (!m.all_default) {
    bw.WriteBool(extra_fields);
    if (extra_fields) {
      bw.WriteBits(3, m.orientation - 1);
      bw.WriteBool(m.have_intr_size);
      if (m.have_intr_size) WriteSizeHeader(bw, m.intrinsic_size);
      bw.WriteBool(m.have_preview);
      if (m.have_preview) WritePreviewHeader(bw, m.preview);
      bw.WriteBool(m.have_animation);
      if (m.have_animation) WriteAnimationHeader(bw, m.animation);
    }
    WriteBitDepth(bw, m.bit_depth);
    bw.WriteBool(m.modular_16bit_buffers);
    WriteU32(bw, m.num_extra, U32Dist::Imm(0), U32Dist::Imm(1),
             U32Dist::BitsOffset(4, 2), U32Dist::BitsOffset(12, 1));
    for (const auto& ec : m.ec_info) WriteExtraChannelInfo(bw, ec);
    bw.WriteBool(m.xyb_encoded);
    WriteColourEncoding(bw, m.colour_encoding);
    if (extra_fields) WriteToneMapping(bw, m.tone_mapping);
    WriteExtensions(bw, m.extensions);
  }

  // CustomTransformData — always written.
  bw.WriteBool(m.default_m);
  if (!m.default_m) {
    if (m.xyb_encoded) WriteOpsinInverseMatrix(bw, m.opsin_inverse_matrix);
    bw.WriteBits(3, m.cw_mask);
    for (uint16_t w : m.up2_weight) bw.WriteF16(w);
    for (uint16_t w : m.up4_weight) bw.WriteF16(w);
    for (uint16_t w : m.up8_weight) bw.WriteF16(w);
  }
}

// ── Public API
// ────────────────────────────────────────────────────────────────

bool ReadImageHeader(BitReader& br, ImageHeader* hdr) {
  {
    const size_t p = br.pos();
    const uint32_t sig = br.ReadBits(16);
    TraceReadField(p, "read.image.signature", "0x%04" PRIX32, sig);
    if (sig != 0x0AFF) {
      fprintf(stderr, "jxltran: invalid JXL codestream signature 0x%04X\n",
              sig);
      return false;
    }
  }
  if (!ReadSizeHeader(br, &hdr->size)) return false;
  if (!ReadImageMetadata(br, hdr)) return false;
  // The encoder pads the image header to a byte boundary; the decoder calls
  // JumpToByteBoundary() here too (lib/jxl/decode.cc).
  {
    const size_t p = br.pos();
    br.ZeroPadToByte();
    TraceReadField(p, "read.image.zero_pad_to_byte", "(end image header)");
  }
  return br.ok();
}

void WriteImageHeader(BitWriter& bw, const ImageHeader& hdr,
                      const uint8_t* original_data) {
  {
    const size_t p = bw.bit_pos();
    bw.WriteBits(16, 0x0AFF);
    TraceWriteField(p, "write.image.signature", "0xFF0A");
  }
  WriteSizeHeader(bw, hdr.size);
  {
    const size_t p = bw.bit_pos();
    WriteImageMetadata(bw, hdr.metadata);
    TraceWriteField(p, "write.image.metadata", "all_default=%s",
                    hdr.metadata.all_default ? "true" : "false");
  }
  // If an ICC profile was present, copy its entropy-coded bits verbatim.
  if (hdr.metadata.colour_encoding.want_icc && original_data) {
    const size_t p = bw.bit_pos();
    bw.AppendBitsRange(original_data, hdr.icc_start_bit,
                       hdr.icc_end_bit - hdr.icc_start_bit);
    TraceWriteField(p, "write.image.icc_entropy", "bits=%" PRIuS "",
                    hdr.icc_end_bit - hdr.icc_start_bit);
  }
  {
    const size_t p = bw.bit_pos();
    bw.ZeroPadToByte();
    TraceWriteField(p, "write.image.zero_pad_to_byte", "(end image header)");
  }
}

}  // namespace jxltran
