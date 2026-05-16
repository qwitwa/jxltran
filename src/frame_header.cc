// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "frame_header.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "codestream.h"
#include "entropy.h"
#include "printf_macros.h"
#include "toc.h"
#include "trace.h"

namespace jxltran {

namespace {

// ── Trace helpers (verbose)
// ───────────────────────────────────────────────────────────────────

// Returns UTF-8 sequence length 1–4, or 0 if invalid / incomplete.
static size_t Utf8Codepoint(const uint8_t* s, size_t len, uint32_t* cp_out) {
  if (len == 0) return 0;
  const uint8_t b0 = s[0];
  if (b0 < 0x80) {
    *cp_out = b0;
    return 1;
  }
  auto cont = [&](size_t i) -> bool {
    return i < len && (s[i] & 0xC0) == 0x80;
  };
  if ((b0 & 0xE0) == 0xC0) {
    if (len < 2 || !cont(1)) return 0;
    uint32_t cp = (static_cast<uint32_t>(b0 & 0x1F) << 6) |
                  static_cast<uint32_t>(s[1] & 0x3F);
    if (cp < 0x80) return 0;
    *cp_out = cp;
    return 2;
  }
  if ((b0 & 0xF0) == 0xE0) {
    if (len < 3 || !cont(1) || !cont(2)) return 0;
    uint32_t cp = (static_cast<uint32_t>(b0 & 0x0F) << 12) |
                  (static_cast<uint32_t>(s[1] & 0x3F) << 6) |
                  static_cast<uint32_t>(s[2] & 0x3F);
    if (cp < 0x800 || (cp >= 0xD800 && cp <= 0xDFFF)) return 0;
    *cp_out = cp;
    return 3;
  }
  if ((b0 & 0xF8) == 0xF0) {
    if (len < 4 || !cont(1) || !cont(2) || !cont(3)) return 0;
    uint32_t cp = (static_cast<uint32_t>(b0 & 0x07) << 18) |
                  (static_cast<uint32_t>(s[1] & 0x3F) << 12) |
                  (static_cast<uint32_t>(s[2] & 0x3F) << 6) |
                  static_cast<uint32_t>(s[3] & 0x3F);
    if (cp < 0x10000 || cp > 0x10FFFF) return 0;
    *cp_out = cp;
    return 4;
  }
  return 0;
}

static std::string CompactNameHex(const std::vector<uint8_t>& name,
                                  size_t max_bytes = 32) {
  std::string out;
  const size_t n = std::min(name.size(), max_bytes);
  out.reserve(n * 2 + 24);
  for (size_t i = 0; i < n; ++i) {
    char buf[4];
    snprintf(buf, sizeof(buf), "%02x", static_cast<unsigned>(name[i]));
    out += buf;
    if (i + 1 < n && ((i & 3u) == 3u)) out += ' ';
  }
  if (name.size() > max_bytes) {
    char tail[48];
    snprintf(tail, sizeof(tail), " …(+%" PRIuS "B)", name.size() - max_bytes);
    out += tail;
  }
  return out;
}

static std::string NameUtf8LossyDisplay(const std::vector<uint8_t>& name,
                                        size_t max_chars = 96) {
  std::string out;
  out.reserve(std::min(name.size(), max_chars) + 8);
  size_t i = 0;
  while (i < name.size() && out.size() < max_chars) {
    uint32_t cp = 0;
    const size_t n = Utf8Codepoint(name.data() + i, name.size() - i, &cp);
    if (n == 0) {
      char esc[8];
      snprintf(esc, sizeof(esc), "\\x%02x", static_cast<unsigned>(name[i]));
      out += esc;
      ++i;
      continue;
    }
    const bool printable =
        (cp >= 32 && cp != 127 && cp < 0x110000 &&
         !(cp >= 0xD800 && cp <= 0xDFFF) && (cp < 0x80 || cp >= 0xA0));
    if (!printable) {
      out += '.';
      i += n;
      continue;
    }
    for (size_t k = 0; k < n; ++k) {
      out += static_cast<char>(name[i + k]);
    }
    i += n;
  }
  if (i < name.size()) out += "…";
  return out;
}

static std::string FormatFrameNameTraceLine(const std::vector<uint8_t>& name) {
  const std::string hex = CompactNameHex(name, 32);
  std::string utf = NameUtf8LossyDisplay(name, 96);
  for (char& c : utf) {
    if (c == '%') c = '?';
  }
  char buf[640];
  snprintf(buf, sizeof(buf), "len=%" PRIuS " hex=[%s] utf8=%s", name.size(),
           hex.c_str(), utf.c_str());
  return std::string(buf);
}

}  // namespace

// ── Utility
// ───────────────────────────────────────────────────────────────────

static inline size_t DivCeil(size_t a, size_t b) { return (a + b - 1) / b; }

// U32 coding used for crop dimensions and offsets (spec §D).
static const U32Dist kCropDist[4] = {
    U32Dist::Bits(8),
    U32Dist::BitsOffset(11, 256),
    U32Dist::BitsOffset(14, 2304),
    U32Dist::BitsOffset(30, 18688),
};
static inline uint32_t ReadCropU32(BitReader& br) {
  return ReadU32(br, kCropDist[0], kCropDist[1], kCropDist[2], kCropDist[3]);
}
static inline void WriteCropU32(BitWriter& bw, uint32_t v) {
  WriteU32(bw, v, kCropDist[0], kCropDist[1], kCropDist[2], kCropDist[3]);
}

// ── Passes
// ────────────────────────────────────────────────────────────────────

static bool ReadPasses(BitReader& br, FramePasses* p) {
  p->num_passes = ReadU32(br, U32Dist::Imm(1), U32Dist::Imm(2), U32Dist::Imm(3),
                          U32Dist::BitsOffset(3, 4));
  if (p->num_passes == 0 || p->num_passes > 8) return false;
  if (p->num_passes != 1) {
    p->num_ds = ReadU32(br, U32Dist::Imm(0), U32Dist::Imm(1), U32Dist::Imm(2),
                        U32Dist::BitsOffset(1, 3));
    if (p->num_ds >= p->num_passes) return false;
    for (uint32_t i = 0; i < p->num_passes - 1; ++i) {
      p->shift[i] = br.ReadBits(2);
    }
    for (uint32_t i = 0; i < p->num_ds; ++i) {
      p->downsample[i] = ReadU32(br, U32Dist::Imm(1), U32Dist::Imm(2),
                                 U32Dist::Imm(4), U32Dist::Imm(8));
    }
    for (uint32_t i = 0; i < p->num_ds; ++i) {
      p->last_pass[i] = ReadU32(br, U32Dist::Imm(0), U32Dist::Imm(1),
                                U32Dist::Imm(2), U32Dist::BitsOffset(3, 0));
    }
  }
  return true;
}

static void WritePasses(BitWriter& bw, const FramePasses& p) {
  WriteU32(bw, p.num_passes, U32Dist::Imm(1), U32Dist::Imm(2), U32Dist::Imm(3),
           U32Dist::BitsOffset(3, 4));
  if (p.num_passes != 1) {
    WriteU32(bw, p.num_ds, U32Dist::Imm(0), U32Dist::Imm(1), U32Dist::Imm(2),
             U32Dist::BitsOffset(1, 3));
    for (uint32_t i = 0; i < p.num_passes - 1; ++i) bw.WriteBits(2, p.shift[i]);
    for (uint32_t i = 0; i < p.num_ds; ++i) {
      WriteU32(bw, p.downsample[i], U32Dist::Imm(1), U32Dist::Imm(2),
               U32Dist::Imm(4), U32Dist::Imm(8));
    }
    for (uint32_t i = 0; i < p.num_ds; ++i) {
      WriteU32(bw, p.last_pass[i], U32Dist::Imm(0), U32Dist::Imm(1),
               U32Dist::Imm(2), U32Dist::BitsOffset(3, 0));
    }
  }
}

// ── BlendingInfo
// ──────────────────────────────────────────────────────────────

// Lib's BlendingInfo serializes `source` when mode != kReplace OR the frame
// is partial relative to the canvas (see FrameHeader::VisitFields).
bool FrameNeedsBlendingSourceField(const FrameHeader& fh, uint32_t canvas_w,
                                   uint32_t canvas_h) {
  const bool normal = (fh.frame_type == kFrameTypeRegular ||
                       fh.frame_type == kFrameTypeSkipProgressive);
  if (!fh.have_crop || !normal) return false;
  const int32_t x0 =
      (fh.frame_type != kFrameTypeReferenceOnly) ? UnpackSigned(fh.ux0) : 0;
  const int32_t y0 =
      (fh.frame_type != kFrameTypeReferenceOnly) ? UnpackSigned(fh.uy0) : 0;
  const int32_t iw = static_cast<int32_t>(canvas_w);
  const int32_t ih = static_cast<int32_t>(canvas_h);
  bool is_partial = false;
  is_partial |= x0 > 0;
  is_partial |= y0 > 0;
  is_partial |= static_cast<int32_t>(fh.crop_width) + x0 < iw;
  is_partial |= static_cast<int32_t>(fh.crop_height) + y0 < ih;
  return is_partial;
}

// resets_canvas depends on blending mode and frame coverage.  It is computed
// externally; we receive it as a parameter to know whether `source` is present.
static void ReadBlendingInfo(BitReader& br, uint32_t num_extra,
                             bool is_partial_frame, FrameBlendingInfo* bi) {
  bi->mode = ReadU32(br, U32Dist::Imm(0), U32Dist::Imm(1), U32Dist::Imm(2),
                     U32Dist::BitsOffset(2, 3));
  bool has_alpha = (num_extra > 0) &&
                   (bi->mode == 2 /* kBlend */ || bi->mode == 3 /* kMulAdd */);
  bool has_clamp = (num_extra > 0) &&
                   (bi->mode == 2 || bi->mode == 3 || bi->mode == 4 /* kMul */);
  if (has_alpha) {
    bi->alpha_channel = ReadU32(br, U32Dist::Imm(0), U32Dist::Imm(1),
                                U32Dist::Imm(2), U32Dist::BitsOffset(3, 3));
  }
  if (has_clamp) bi->clamp = br.ReadBool();
  if (bi->mode != 0 || is_partial_frame) {
    bi->source = br.ReadBits(2);
  }
}

static void WriteBlendingInfo(BitWriter& bw, const FrameBlendingInfo& bi,
                              uint32_t num_extra, bool is_partial_frame) {
  WriteU32(bw, bi.mode, U32Dist::Imm(0), U32Dist::Imm(1), U32Dist::Imm(2),
           U32Dist::BitsOffset(2, 3));
  bool has_alpha = (num_extra > 0) && (bi.mode == 2 || bi.mode == 3);
  bool has_clamp =
      (num_extra > 0) && (bi.mode == 2 || bi.mode == 3 || bi.mode == 4);
  if (has_alpha) {
    WriteU32(bw, bi.alpha_channel, U32Dist::Imm(0), U32Dist::Imm(1),
             U32Dist::Imm(2), U32Dist::BitsOffset(3, 3));
  }
  if (has_clamp) bw.WriteBool(bi.clamp);
  if (bi.mode != 0 || is_partial_frame) bw.WriteBits(2, bi.source);
}

// ── FrameHeader read/write
// ────────────────────────────────────────────────────

// Compute whether `full_frame` (frame exactly covers canvas) is true.
static bool ComputeFullFrame(const FrameHeader& fh, uint32_t canvas_w,
                             uint32_t canvas_h) {
  if (!fh.have_crop) return true;
  uint32_t fw = fh.crop_width;
  uint32_t fh_sz = fh.crop_height;
  int32_t x0 =
      (fh.frame_type != kFrameTypeReferenceOnly) ? UnpackSigned(fh.ux0) : 0;
  int32_t y0 =
      (fh.frame_type != kFrameTypeReferenceOnly) ? UnpackSigned(fh.uy0) : 0;
  return x0 <= 0 && y0 <= 0 &&
         x0 + static_cast<int32_t>(fw) >= static_cast<int32_t>(canvas_w) &&
         y0 + static_cast<int32_t>(fh_sz) >= static_cast<int32_t>(canvas_h);
}

// Skip FrameHeader-level extension payloads (after reading the U64 mask).
static void SkipFrameHeaderExtensionPayloads(BitReader& br, uint64_t mask) {
  if (mask == 0) return;
  uint64_t rem = mask;
  std::vector<uint64_t> counts;
  while (rem) {
    counts.push_back(ReadU64(br));
    rem &= rem - 1;
  }
  for (uint64_t bc : counts) {
    for (uint64_t i = 0; i < bc; ++i) br.ReadBits(1);
  }
}

// Implicit defaults after a packed (single-bit) frame header (lib Bundle::SetDefault
// for a regular VarDCT still image).
static void MaterializeAllDefaultFrameHeader(FrameHeader* fh,
                                             const ImageMetadata& meta,
                                             uint32_t /*canvas_w*/,
                                             uint32_t /*canvas_h*/) {
  *fh = FrameHeader{};
  fh->all_default = false;
  fh->frame_type = kFrameTypeRegular;
  fh->encoding = kFrameEncVarDCT;
  fh->flags = 0;
  fh->do_YCbCr = false;
  fh->jpeg_upsampling[0] = fh->jpeg_upsampling[1] = fh->jpeg_upsampling[2] = 0;
  fh->upsampling = 1;
  fh->ec_upsampling.assign(meta.num_extra, 1u);
  fh->group_size_shift = 1;
  fh->x_qm_scale = 3;
  fh->b_qm_scale = 2;
  fh->passes = FramePasses{};
  fh->lf_level = 0;
  fh->have_crop = false;
  fh->ux0 = 0;
  fh->uy0 = 0;
  fh->crop_width = 0;
  fh->crop_height = 0;
  fh->blending_info = FrameBlendingInfo{};
  fh->ec_blending_info.assign(meta.num_extra, FrameBlendingInfo{});
  fh->duration = 0;
  fh->timecode = 0;
  fh->is_last = true;
  fh->save_as_reference = 0;
  fh->save_before_ct = false;
  fh->save_before_ct_present = false;
  fh->name.clear();
  fh->restoration = ParsedRestorationFilter{};
  fh->restoration.all_default = true;
  fh->rf_reencode = false;
  fh->rf_start_bit = 0;
  fh->frame_extensions_start_bit = 0;
  fh->rf_ext_end_bit = 0;
  fh->frame_extensions_mask = 0;
  fh->crop_bundle_start_bit = 0;
}

static bool FrameHeaderEqualsMaterializedDefault(
    const FrameHeader& fh, const ImageMetadata& meta, uint32_t canvas_w,
    uint32_t canvas_h) {
  FrameHeader r;
  MaterializeAllDefaultFrameHeader(&r, meta, canvas_w, canvas_h);
  if (fh.rf_reencode != r.rf_reencode) return false;
  if (fh.restoration.all_default != r.restoration.all_default) return false;
  if (fh.frame_extensions_mask != r.frame_extensions_mask) return false;
  if (fh.frame_type != r.frame_type) return false;
  if (fh.encoding != r.encoding) return false;
  if (fh.flags != r.flags) return false;
  if (fh.do_YCbCr != r.do_YCbCr) return false;
  if (fh.jpeg_upsampling[0] != r.jpeg_upsampling[0] ||
      fh.jpeg_upsampling[1] != r.jpeg_upsampling[1] ||
      fh.jpeg_upsampling[2] != r.jpeg_upsampling[2]) {
    return false;
  }
  if (fh.upsampling != r.upsampling) return false;
  if (fh.ec_upsampling != r.ec_upsampling) return false;
  if (fh.group_size_shift != r.group_size_shift) return false;
  if (fh.x_qm_scale != r.x_qm_scale || fh.b_qm_scale != r.b_qm_scale) {
    return false;
  }
  if (fh.passes.num_passes != r.passes.num_passes ||
      fh.passes.num_ds != r.passes.num_ds) {
    return false;
  }
  if (fh.passes.num_passes != 1) {
    for (int i = 0; i < 8; ++i) {
      if (fh.passes.shift[i] != r.passes.shift[i] ||
          fh.passes.downsample[i] != r.passes.downsample[i] ||
          fh.passes.last_pass[i] != r.passes.last_pass[i]) {
        return false;
      }
    }
  }
  if (fh.lf_level != r.lf_level) return false;
  if (fh.have_crop != r.have_crop || fh.ux0 != r.ux0 || fh.uy0 != r.uy0 ||
      fh.crop_width != r.crop_width || fh.crop_height != r.crop_height) {
    return false;
  }
  if (fh.blending_info.mode != r.blending_info.mode ||
      fh.blending_info.alpha_channel != r.blending_info.alpha_channel ||
      fh.blending_info.clamp != r.blending_info.clamp ||
      fh.blending_info.source != r.blending_info.source) {
    return false;
  }
  if (fh.ec_blending_info.size() != r.ec_blending_info.size()) return false;
  for (size_t i = 0; i < fh.ec_blending_info.size(); ++i) {
    const FrameBlendingInfo& a = fh.ec_blending_info[i];
    const FrameBlendingInfo& b = r.ec_blending_info[i];
    if (a.mode != b.mode || a.alpha_channel != b.alpha_channel ||
        a.clamp != b.clamp || a.source != b.source) {
      return false;
    }
  }
  if (fh.duration != r.duration || fh.timecode != r.timecode) return false;
  if (fh.is_last != r.is_last) return false;
  if (fh.save_as_reference != r.save_as_reference) return false;
  if (fh.save_before_ct_present != r.save_before_ct_present ||
      fh.save_before_ct != r.save_before_ct) {
    return false;
  }
  if (fh.name != r.name) return false;
  (void)canvas_w;
  (void)canvas_h;
  return true;
}

static bool ComputeFrameHeaderPacksAsOneDefaultBit(
    const FrameHeader& fh, const ImageMetadata& meta, uint32_t canvas_w,
    uint32_t canvas_h) {
  return FrameHeaderEqualsMaterializedDefault(fh, meta, canvas_w, canvas_h);
}

bool ReadFrameHeader(BitReader& br, const ImageMetadata& meta,
                     uint32_t canvas_w, uint32_t canvas_h, FrameHeader* fh,
                     const uint8_t* original_data) {
  (void)original_data;
  const size_t p_all = br.pos();
  const bool packed_all_default = br.ReadBool();
  TraceReadField(p_all, "read.frame.all_default", "%s",
                 packed_all_default ? "true" : "false");
  if (packed_all_default) {
    MaterializeAllDefaultFrameHeader(fh, meta, canvas_w, canvas_h);
    fh->rf_reencode = false;
    return br.ok();
  }

  fh->all_default = false;
  fh->crop_bundle_start_bit = 0;

  {
    const size_t p = br.pos();
    fh->frame_type = br.ReadBits(2);
    TraceReadField(p, "read.frame.frame_type", "%u", fh->frame_type);
  }
  {
    const size_t p = br.pos();
    fh->encoding = br.ReadBits(1);
    TraceReadField(p, "read.frame.encoding", "%u", fh->encoding);
  }
  {
    const size_t p = br.pos();
    fh->flags = ReadU64(br);
    TraceReadField(p, "read.frame.flags", "%" PRIu64 "", fh->flags);
  }

  const bool use_lf_frame = (fh->flags & kFlagUseLfFrame) != 0;
  const bool normal_frame = (fh->frame_type == kFrameTypeRegular ||
                             fh->frame_type == kFrameTypeSkipProgressive);

  if (!meta.xyb_encoded) {
    const size_t p = br.pos();
    fh->do_YCbCr = br.ReadBool();
    TraceReadField(p, "read.frame.do_YCbCr", "%s",
                   fh->do_YCbCr ? "true" : "false");
  }
  if (fh->do_YCbCr && !use_lf_frame) {
    for (int c = 0; c < 3; ++c) {
      const size_t p = br.pos();
      fh->jpeg_upsampling[c] = br.ReadBits(2);
      TraceReadField(p, "read.frame.jpeg_upsampling[]", "c=%d val=%u", c,
                     fh->jpeg_upsampling[c]);
    }
  }
  if (!use_lf_frame) {
    {
      const size_t p = br.pos();
      fh->upsampling = ReadU32(br, U32Dist::Imm(1), U32Dist::Imm(2),
                               U32Dist::Imm(4), U32Dist::Imm(8));
      TraceReadField(p, "read.frame.upsampling", "%u", fh->upsampling);
    }
    fh->ec_upsampling.resize(meta.num_extra, 1);
    for (uint32_t i = 0; i < meta.num_extra; ++i) {
      const size_t p = br.pos();
      fh->ec_upsampling[i] = ReadU32(br, U32Dist::Imm(1), U32Dist::Imm(2),
                                     U32Dist::Imm(4), U32Dist::Imm(8));
      TraceReadField(p, "read.frame.ec_upsampling[]", "i=%" PRIu32 " val=%u",
                     i, fh->ec_upsampling[i]);
    }
  }
  if (fh->encoding == kFrameEncModular) {
    const size_t p = br.pos();
    fh->group_size_shift = br.ReadBits(2);
    TraceReadField(p, "read.frame.group_size_shift", "%u",
                   fh->group_size_shift);
  }
  if (meta.xyb_encoded && fh->encoding == kFrameEncVarDCT) {
    {
      const size_t p = br.pos();
      fh->x_qm_scale = br.ReadBits(3);
      TraceReadField(p, "read.frame.x_qm_scale", "%u", fh->x_qm_scale);
    }
    {
      const size_t p = br.pos();
      fh->b_qm_scale = br.ReadBits(3);
      TraceReadField(p, "read.frame.b_qm_scale", "%u", fh->b_qm_scale);
    }
  }
  if (fh->frame_type != kFrameTypeReferenceOnly) {
    const size_t pass_p = br.pos();
    if (!ReadPasses(br, &fh->passes)) return false;
    TraceReadField(pass_p, "read.frame.passes", "num_passes=%" PRIu32
                                                " num_ds=%" PRIu32 "",
                   fh->passes.num_passes, fh->passes.num_ds);
  }

  if (fh->frame_type == kFrameTypeLF) {
    const size_t p = br.pos();
    fh->lf_level = 1 + br.ReadBits(2);
    TraceReadField(p, "read.frame.lf_level", "%u", fh->lf_level);
  }

  if (fh->frame_type != kFrameTypeLF) {
    fh->crop_bundle_start_bit = br.pos();
    TraceReadField(fh->crop_bundle_start_bit, "read.frame.crop_bundle",
                   "(have_crop starts)");
    {
      const size_t p = br.pos();
      fh->have_crop = br.ReadBool();
      TraceReadField(p, "read.frame.have_crop", "%s",
                     fh->have_crop ? "true" : "false");
    }
    if (fh->have_crop) {
      if (fh->frame_type != kFrameTypeReferenceOnly) {
        {
          const size_t p = br.pos();
          fh->ux0 = ReadCropU32(br);
          TraceReadField(p, "read.frame.ux0",
                         "%" PRIu32 " (x0 = %" PRId32 ")", fh->ux0,
                         UnpackSigned(fh->ux0));
        }
        {
          const size_t p = br.pos();
          fh->uy0 = ReadCropU32(br);
          TraceReadField(p, "read.frame.uy0",
                         "%" PRIu32 " (y0 = %" PRId32 ")", fh->uy0,
                         UnpackSigned(fh->uy0));
        }
      }
      {
        const size_t p = br.pos();
        fh->crop_width = ReadCropU32(br);
        TraceReadField(p, "read.frame.crop_width", "%" PRIu32 "",
                       fh->crop_width);
      }
      {
        const size_t p = br.pos();
        fh->crop_height = ReadCropU32(br);
        TraceReadField(p, "read.frame.crop_height", "%" PRIu32 "",
                       fh->crop_height);
      }
    }
  }

  // Compute resets_canvas using the current canvas dims.
  auto ComputeRC = [&]() -> bool {
    return ComputeFullFrame(*fh, canvas_w, canvas_h) &&
           (fh->blending_info.mode == 0);
  };

  if (normal_frame) {
    const bool is_partial = FrameNeedsBlendingSourceField(*fh, canvas_w, canvas_h);
    {
      const size_t p = br.pos();
      ReadBlendingInfo(br, meta.num_extra, is_partial, &fh->blending_info);
      TraceReadField(p, "read.frame.blending_info", "mode=%" PRIu32 "",
                     fh->blending_info.mode);
    }
    fh->ec_blending_info.resize(meta.num_extra);
    for (uint32_t i = 0; i < meta.num_extra; ++i) {
      const size_t p = br.pos();
      ReadBlendingInfo(br, meta.num_extra, is_partial, &fh->ec_blending_info[i]);
      TraceReadField(p, "read.frame.ec_blending_info[]", "i=%" PRIu32
                                                          " mode=%" PRIu32 "",
                     i, fh->ec_blending_info[i].mode);
    }
    if (meta.have_animation) {
      const size_t p = br.pos();
      fh->duration = ReadU32(br, U32Dist::Imm(0), U32Dist::Imm(1),
                             U32Dist::Bits(8), U32Dist::Bits(32));
      TraceReadField(p, "read.frame.duration", "%" PRIu32 "", fh->duration);
    }
    if (meta.animation.have_timecodes) {
      const size_t p = br.pos();
      fh->timecode = br.ReadBits(32);
      TraceReadField(p, "read.frame.timecode", "%" PRIu32 "", fh->timecode);
    }
    {
      const size_t p = br.pos();
      fh->is_last = br.ReadBool();
      TraceReadField(p, "read.frame.is_last", "%s",
                     fh->is_last ? "true" : "false");
    }
  } else {
    // kLFFrame or kReferenceOnly — is_last not present.
    // Default: !frame_type.  For kLFFrame(1) and kReferenceOnly(2), !x = false.
    fh->is_last = false;
  }

  if (fh->frame_type != kFrameTypeLF && !fh->is_last) {
    const size_t p = br.pos();
    fh->save_as_reference = br.ReadBits(2);
    TraceReadField(p, "read.frame.save_as_reference", "%" PRIu32 "",
                   fh->save_as_reference);
  }

  {
    bool resets_canvas = ComputeRC();
    bool can_reference = !fh->is_last && fh->frame_type != kFrameTypeLF &&
                         (fh->duration == 0 || fh->save_as_reference != 0);
    fh->save_before_ct_present = (fh->frame_type == kFrameTypeReferenceOnly) ||
                                 (resets_canvas && can_reference);
    if (fh->save_before_ct_present) {
      const size_t p = br.pos();
      fh->save_before_ct = br.ReadBool();
      TraceReadField(p, "read.frame.save_before_ct", "%s",
                     fh->save_before_ct ? "true" : "false");
    } else {
      fh->save_before_ct = !normal_frame;
    }
  }

  // name_len + name[name_len]
  uint32_t name_len;
  {
    const size_t p = br.pos();
    name_len = ReadU32(br, U32Dist::Imm(0), U32Dist::Bits(4),
                       U32Dist::BitsOffset(5, 16), U32Dist::BitsOffset(10, 48));
    TraceReadField(p, "read.frame.name_len", "%" PRIu32 "", name_len);
  }
  const size_t name_body_start = br.pos();
  fh->name.resize(name_len);
  for (uint32_t i = 0; i < name_len; ++i) {
    fh->name[i] = static_cast<uint8_t>(br.ReadBits(8));
  }
  if (TraceIsOn()) {
    if (name_len == 0) {
      TraceReadField(name_body_start, "read.frame.name", "len=0");
    } else {
      const std::string line = FormatFrameNameTraceLine(fh->name);
      TraceReadField(name_body_start, "read.frame.name", "%s", line.c_str());
    }
  }

  // RestorationFilter + Extensions
  fh->rf_start_bit = br.pos();
  TraceReadField(fh->rf_start_bit, "read.frame.restoration_filter",
                 "(loop_filter+its_extensions start)");
  fh->rf_reencode = false;
  const bool modular = (fh->encoding == kFrameEncModular);
  if (!ReadParsedRestorationFilter(br, modular, &fh->restoration)) {
    return false;
  }
  fh->frame_extensions_start_bit = br.pos();
  TraceReadField(fh->frame_extensions_start_bit,
                 "read.frame.frame_header_extensions", "(start)");
  fh->frame_extensions_mask = ReadU64(br);
  SkipFrameHeaderExtensionPayloads(br, fh->frame_extensions_mask);
  fh->rf_ext_end_bit = br.pos();
  TraceReadField(fh->rf_ext_end_bit, "read.frame.frame_header_extensions_end",
                 "span_bits=%" PRIuS "", fh->rf_ext_end_bit - fh->rf_start_bit);

  return br.ok();
}

bool WriteFrameHeaderFromHaveCrop(BitWriter& bw, const FrameHeader& fh,
                                  const ImageMetadata& meta,
                                  const uint8_t* original_data,
                                  uint32_t canvas_w, uint32_t canvas_h) {
  const bool normal_frame = (fh.frame_type == kFrameTypeRegular ||
                             fh.frame_type == kFrameTypeSkipProgressive);

  if (fh.frame_type != kFrameTypeLF) {
    {
      const size_t p = bw.bit_pos();
      bw.WriteBool(fh.have_crop);
      TraceWriteField(p, "write.frame_tail.have_crop", "%s",
                      fh.have_crop ? "true" : "false");
    }
    if (fh.have_crop) {
      if (fh.frame_type != kFrameTypeReferenceOnly) {
        {
          const size_t p = bw.bit_pos();
          WriteCropU32(bw, fh.ux0);
          TraceWriteField(p, "write.frame_tail.ux0",
                          "%" PRIu32 " (x0 = %" PRId32 ")", fh.ux0,
                          UnpackSigned(fh.ux0));
        }
        {
          const size_t p = bw.bit_pos();
          WriteCropU32(bw, fh.uy0);
          TraceWriteField(p, "write.frame_tail.uy0",
                          "%" PRIu32 " (y0 = %" PRId32 ")", fh.uy0,
                          UnpackSigned(fh.uy0));
        }
      }
      {
        const size_t p = bw.bit_pos();
        WriteCropU32(bw, fh.crop_width);
        TraceWriteField(p, "write.frame_tail.crop_width", "%" PRIu32 "",
                        fh.crop_width);
      }
      {
        const size_t p = bw.bit_pos();
        WriteCropU32(bw, fh.crop_height);
        TraceWriteField(p, "write.frame_tail.crop_height", "%" PRIu32 "",
                        fh.crop_height);
      }
    }
  }

  if (normal_frame) {
    const bool is_partial = FrameNeedsBlendingSourceField(fh, canvas_w, canvas_h);
    {
      const size_t p = bw.bit_pos();
      WriteBlendingInfo(bw, fh.blending_info, meta.num_extra, is_partial);
      TraceWriteField(p, "write.frame_tail.blending_info", "mode=%" PRIu32 "",
                      fh.blending_info.mode);
    }
    for (uint32_t i = 0; i < meta.num_extra; ++i) {
      const FrameBlendingInfo& ec = i < fh.ec_blending_info.size()
                                        ? fh.ec_blending_info[i]
                                        : fh.blending_info;
      const size_t p = bw.bit_pos();
      WriteBlendingInfo(bw, ec, meta.num_extra, is_partial);
      TraceWriteField(p, "write.frame_tail.ec_blending_info[]",
                      "i=%" PRIu32 " mode=%" PRIu32 "", i, ec.mode);
    }
    if (meta.have_animation) {
      const size_t p = bw.bit_pos();
      WriteU32(bw, fh.duration, U32Dist::Imm(0), U32Dist::Imm(1),
               U32Dist::Bits(8), U32Dist::Bits(32));
      TraceWriteField(p, "write.frame_tail.duration", "%" PRIu32 "",
                      fh.duration);
    }
    if (meta.animation.have_timecodes) {
      const size_t p = bw.bit_pos();
      bw.WriteBits(32, fh.timecode);
      TraceWriteField(p, "write.frame_tail.timecode", "%" PRIu32 "",
                      fh.timecode);
    }
    {
      const size_t p = bw.bit_pos();
      bw.WriteBool(fh.is_last);
      TraceWriteField(p, "write.frame_tail.is_last", "%s",
                      fh.is_last ? "true" : "false");
    }
  }

  if (fh.frame_type != kFrameTypeLF && !fh.is_last) {
    const size_t p = bw.bit_pos();
    bw.WriteBits(2, fh.save_as_reference);
    TraceWriteField(p, "write.frame_tail.save_as_reference", "%" PRIu32 "",
                    fh.save_as_reference);
  }

  // save_before_ct: condition must match read-time layout (lib uses metadata
  // canvas size for full-frame / resets_canvas, same as |canvas_w|/|h|).
  {
    bool full_frame = ComputeFullFrame(fh, canvas_w, canvas_h);
    bool resets_canvas = full_frame && (fh.blending_info.mode == 0);
    bool can_reference = !fh.is_last && fh.frame_type != kFrameTypeLF &&
                         (fh.duration == 0 || fh.save_as_reference != 0);
    bool sbc_present = (fh.frame_type == kFrameTypeReferenceOnly) ||
                       (resets_canvas && can_reference);
    if (sbc_present) {
      const size_t p = bw.bit_pos();
      bw.WriteBool(fh.save_before_ct);
      TraceWriteField(p, "write.frame_tail.save_before_ct", "%s",
                      fh.save_before_ct ? "true" : "false");
    }
  }

  // name
  {
    const size_t p = bw.bit_pos();
    WriteU32(bw, static_cast<uint32_t>(fh.name.size()), U32Dist::Imm(0),
             U32Dist::Bits(4), U32Dist::BitsOffset(5, 16),
             U32Dist::BitsOffset(10, 48));
    TraceWriteField(p, "write.frame_tail.name_len", "%" PRIuS "",
                    fh.name.size());
  }
  const size_t name_body_start = bw.bit_pos();
  for (size_t i = 0; i < fh.name.size(); ++i) {
    bw.WriteBits(8, fh.name[i]);
  }
  if (TraceIsOn()) {
    if (fh.name.empty()) {
      TraceWriteField(name_body_start, "write.frame_tail.name", "len=0");
    } else {
      const std::string line = FormatFrameNameTraceLine(fh.name);
      TraceWriteField(name_body_start, "write.frame_tail.name", "%s",
                      line.c_str());
    }
  }

  // Loop filter bundle, then frame-header extensions (verbatim or re-encoded LF
  // + verbatim suffix).
  if (fh.rf_reencode) {
    const size_t p = bw.bit_pos();
    const bool modular = (fh.encoding == kFrameEncModular);
    if (!WriteParsedRestorationFilter(bw, modular, fh.restoration)) {
      return false;
    }
    TraceWriteField(p, "write.frame_tail.rf_reencode",
                    "loop_filter_bits=%" PRIuS "", bw.bit_pos() - p);
    if (fh.frame_extensions_start_bit < fh.rf_ext_end_bit) {
      const size_t q = bw.bit_pos();
      bw.AppendBitsRange(original_data, fh.frame_extensions_start_bit,
                           fh.rf_ext_end_bit - fh.frame_extensions_start_bit);
      TraceWriteField(q, "write.frame_tail.frame_header_extensions_verbatim",
                      "span_bits=%" PRIuS "",
                      fh.rf_ext_end_bit - fh.frame_extensions_start_bit);
    }
  } else if (original_data != nullptr && fh.rf_start_bit != fh.rf_ext_end_bit) {
    const size_t p = bw.bit_pos();
    bw.AppendBitsRange(original_data, fh.rf_start_bit,
                       fh.rf_ext_end_bit - fh.rf_start_bit);
    TraceWriteField(p, "write.frame_tail.rf_and_extensions_verbatim",
                    "span_bits=%" PRIuS "", fh.rf_ext_end_bit - fh.rf_start_bit);
  } else {
    // Packed read (no rf_start..rf_ext range in the source) or missing
    // original: emit loop filter + frame-level extensions from parsed fields.
    // A single WriteBool(true) is not enough — ReadFrameHeader always reads
    // frame_extensions_mask (U64) after the restoration filter bundle.
    const bool modular = (fh.encoding == kFrameEncModular);
    const size_t p = bw.bit_pos();
    if (!WriteParsedRestorationFilter(bw, modular, fh.restoration)) {
      return false;
    }
    TraceWriteField(p, "write.frame_tail.restoration_filter",
                    "span_bits=%" PRIuS "", bw.bit_pos() - p);
    {
      const size_t q = bw.bit_pos();
      WriteU64(bw, fh.frame_extensions_mask);
      TraceWriteField(q, "write.frame_tail.frame_extensions_mask",
                      "mask=0x%" PRIx64 "", fh.frame_extensions_mask);
    }
    if (fh.frame_extensions_mask != 0) {
      fprintf(stderr,
              "jxltran: frame header extensions (mask non-zero) require "
              "verbatim source bits; re-encode from parsed extensions is not "
              "implemented\n");
      return false;
    }
  }
  return true;
}

bool WriteFrameHeader(BitWriter& bw, const FrameHeader& fh,
                      const ImageMetadata& meta, const uint8_t* original_data,
                      uint32_t canvas_w, uint32_t canvas_h) {
  const bool pack =
      ComputeFrameHeaderPacksAsOneDefaultBit(fh, meta, canvas_w, canvas_h);
  {
    const size_t p = bw.bit_pos();
    bw.WriteBool(pack);
    TraceWriteField(p, "write.frame.all_default", "%s",
                    pack ? "true" : "false");
  }
  if (pack) {
    return true;
  }

  {
    const size_t p = bw.bit_pos();
    bw.WriteBits(2, fh.frame_type);
    TraceWriteField(p, "write.frame.frame_type", "%u", fh.frame_type);
  }
  {
    const size_t p = bw.bit_pos();
    bw.WriteBits(1, fh.encoding);
    TraceWriteField(p, "write.frame.encoding", "%u", fh.encoding);
  }
  {
    const size_t p = bw.bit_pos();
    WriteU64(bw, fh.flags);
    TraceWriteField(p, "write.frame.flags", "%" PRIu64 "", fh.flags);
  }

  const bool use_lf_frame = (fh.flags & kFlagUseLfFrame) != 0;

  if (!meta.xyb_encoded) {
    const size_t p = bw.bit_pos();
    bw.WriteBool(fh.do_YCbCr);
    TraceWriteField(p, "write.frame.do_YCbCr", "%s",
                    fh.do_YCbCr ? "true" : "false");
  }
  if (fh.do_YCbCr && !use_lf_frame) {
    for (int c = 0; c < 3; ++c) {
      const size_t p = bw.bit_pos();
      bw.WriteBits(2, fh.jpeg_upsampling[c]);
      TraceWriteField(p, "write.frame.jpeg_upsampling[]", "c=%d val=%u", c,
                      fh.jpeg_upsampling[c]);
    }
  }
  if (!use_lf_frame) {
    {
      const size_t p = bw.bit_pos();
      WriteU32(bw, fh.upsampling, U32Dist::Imm(1), U32Dist::Imm(2),
               U32Dist::Imm(4), U32Dist::Imm(8));
      TraceWriteField(p, "write.frame.upsampling", "%u", fh.upsampling);
    }
    for (uint32_t i = 0; i < meta.num_extra; ++i) {
      uint32_t eu = i < fh.ec_upsampling.size() ? fh.ec_upsampling[i] : 1u;
      const size_t p = bw.bit_pos();
      WriteU32(bw, eu, U32Dist::Imm(1), U32Dist::Imm(2), U32Dist::Imm(4),
               U32Dist::Imm(8));
      TraceWriteField(p, "write.frame.ec_upsampling[]", "i=%" PRIu32 " val=%u",
                      i, eu);
    }
  }
  if (fh.encoding == kFrameEncModular) {
    const size_t p = bw.bit_pos();
    bw.WriteBits(2, fh.group_size_shift);
    TraceWriteField(p, "write.frame.group_size_shift", "%u",
                    fh.group_size_shift);
  }
  if (meta.xyb_encoded && fh.encoding == kFrameEncVarDCT) {
    {
      const size_t p = bw.bit_pos();
      bw.WriteBits(3, fh.x_qm_scale);
      TraceWriteField(p, "write.frame.x_qm_scale", "%u", fh.x_qm_scale);
    }
    {
      const size_t p = bw.bit_pos();
      bw.WriteBits(3, fh.b_qm_scale);
      TraceWriteField(p, "write.frame.b_qm_scale", "%u", fh.b_qm_scale);
    }
  }
  if (fh.frame_type != kFrameTypeReferenceOnly) {
    const size_t p = bw.bit_pos();
    WritePasses(bw, fh.passes);
    TraceWriteField(p, "write.frame.passes", "num_passes=%" PRIu32
                                              " num_ds=%" PRIu32 "",
                    fh.passes.num_passes, fh.passes.num_ds);
  }
  if (fh.frame_type == kFrameTypeLF) {
    const size_t p = bw.bit_pos();
    bw.WriteBits(2, fh.lf_level - 1);
    TraceWriteField(p, "write.frame.lf_level_minus1", "%u", fh.lf_level - 1);
  }
  if (!WriteFrameHeaderFromHaveCrop(bw, fh, meta, original_data, canvas_w,
                                    canvas_h)) {
    return false;
  }
  return true;
}

// ── NumTocEntries (matches lib/jxl/frame_dimensions.h + toc.h) ───────────

// lib/jxl/frame_header.cc — YCbCrChromaSubsampling::kHShift / kVShift.
static constexpr uint8_t kYCbCrSubsamplingH[4] = {0, 1, 1, 0};
static constexpr uint8_t kYCbCrSubsamplingV[4] = {0, 1, 0, 1};

static void ChromaMaxShiftFromJpegModes(const uint32_t mode[3], uint8_t* max_h,
                                        uint8_t* max_v) {
  uint8_t mh = 0, mv = 0;
  for (int c = 0; c < 3; ++c) {
    uint32_t m = mode[c];
    if (m > 3) m = 0;
    mh = static_cast<uint8_t>(
        std::max<unsigned>(mh, kYCbCrSubsamplingH[m]));
    mv = static_cast<uint8_t>(
        std::max<unsigned>(mv, kYCbCrSubsamplingV[m]));
  }
  *max_h = mh;
  *max_v = mv;
}

static constexpr size_t kFdBlockDim = 8;
static constexpr size_t kFdGroupDim = 256;

static size_t AcGroupIndex(size_t pass, size_t group, size_t num_groups,
                           size_t num_dc_groups) {
  return 2 + num_dc_groups + pass * num_groups + group;
}

static size_t NumTocEntriesFromGroups(size_t num_groups, size_t num_dc_groups,
                                      size_t num_passes) {
  if (num_groups == 1 && num_passes == 1) return 1;
  return AcGroupIndex(0, 0, num_groups, num_dc_groups) + num_groups * num_passes;
}

void ComputeFrameTocMetrics(const FrameHeader& fh, const ImageMetadata& meta,
                            uint32_t canvas_w, uint32_t canvas_h,
                            FrameTocMetrics* out) {
  size_t xsize_px = fh.have_crop ? fh.crop_width : canvas_w;
  size_t ysize_px = fh.have_crop ? fh.crop_height : canvas_h;
  if (fh.frame_type == kFrameTypeLF && fh.lf_level > 0) {
    const uint32_t shift = 3 * fh.lf_level;
    xsize_px = DivCeil(xsize_px, static_cast<size_t>(1) << shift);
    ysize_px = DivCeil(ysize_px, static_cast<size_t>(1) << shift);
  }
  const uint32_t up = fh.upsampling > 0 ? fh.upsampling : 1;
  uint32_t gs = fh.group_size_shift;
  if (gs > 3) gs = 3;
  uint8_t max_h = 0, max_v = 0;
  const bool use_lf_frame = (fh.flags & kFlagUseLfFrame) != 0;
  if (!meta.xyb_encoded && fh.do_YCbCr && !use_lf_frame) {
    ChromaMaxShiftFromJpegModes(fh.jpeg_upsampling, &max_h, &max_v);
  }

  out->group_dim = (kFdGroupDim >> 1) << gs;
  out->xsize = DivCeil(xsize_px, static_cast<size_t>(up));
  out->ysize = DivCeil(ysize_px, static_cast<size_t>(up));
  out->xsize_blocks =
      DivCeil(out->xsize, static_cast<size_t>(kFdBlockDim << max_h)) << max_h;
  out->ysize_blocks =
      DivCeil(out->ysize, static_cast<size_t>(kFdBlockDim << max_v)) << max_v;
  out->xsize_groups = DivCeil(out->xsize, out->group_dim);
  out->ysize_groups = DivCeil(out->ysize, out->group_dim);
  out->xsize_dc_groups = DivCeil(out->xsize_blocks, out->group_dim);
  out->ysize_dc_groups = DivCeil(out->ysize_blocks, out->group_dim);
  out->num_groups = out->xsize_groups * out->ysize_groups;
  out->num_dc_groups = out->xsize_dc_groups * out->ysize_dc_groups;
  out->num_passes = fh.passes.num_passes;
  out->all_in_one_section =
      (out->num_groups == 1 && out->num_passes == 1);
  out->num_toc_entries =
      NumTocEntriesFromGroups(out->num_groups, out->num_dc_groups, out->num_passes);
}

// Computes the number of TOC entries for a frame (must match libjxl
// FrameHeader::ToFrameDimensions + NumTocEntries).
static size_t ComputeNumTocEntries(const FrameHeader& fh,
                                   const ImageMetadata& meta, uint32_t canvas_w,
                                   uint32_t canvas_h) {
  FrameTocMetrics m;
  ComputeFrameTocMetrics(fh, meta, canvas_w, canvas_h, &m);
  return m.num_toc_entries;
}

bool ReadFrameTOC(BitReader& br, const FrameHeader& fh,
                  const ImageMetadata& meta, uint32_t canvas_w, uint32_t canvas_h,
                  const uint8_t* frame_base, size_t frame_len,
                  uint64_t* frame_data_bytes, FramedUnit* fu) {
  size_t n = ComputeNumTocEntries(fh, meta, canvas_w, canvas_h);
  if (n == 0) {
    *frame_data_bytes = 0;
    if (fu != nullptr) {
      fu->toc_bit_length = 0;
      fu->toc_decoded_sizes.clear();
      fu->toc_perm.clear();
      fu->toc_bits_before_sizes = 0;
    }
    return true;
  }
  const size_t toc_p = br.pos();
  size_t toc_bits = 0;
  uint64_t total = 0;
  {
    BitReader jr(frame_base, frame_len);
    if (toc_p > jr.size_bits()) {
      return false;
    }
    jr.SkipBits(toc_p);
    std::vector<uint32_t> sizes;
    std::vector<uint32_t> perm;
    if (!ReadFullToc(jr, n, &total, &sizes, &perm,
                      fu != nullptr ? &fu->toc_bits_before_sizes : nullptr)) {
      return false;
    }
    toc_bits = jr.pos() - toc_p;
    *frame_data_bytes = total;
    if (fu != nullptr) {
      fu->toc_decoded_sizes = std::move(sizes);
      if (!perm.empty()) {
        std::vector<uint32_t> logical_at_stream;
        if (!TocPermLogicalAtStreamFromLehmerNaturalToStream(perm, n,
                                                             &logical_at_stream)) {
          return false;
        }
        fu->toc_perm = std::move(logical_at_stream);
      } else {
        fu->toc_perm.clear();
      }
      fu->toc_bit_length = toc_bits;
    }
  }
  br.SkipBits(toc_bits);
  if (!br.ok()) {
    return false;
  }
  TraceReadField(toc_p, "read.frame.toc",
                 "entries=%" PRIuS " frame_data_bytes=%" PRIu64 "", n,
                 *frame_data_bytes);
  return true;
}

}  // namespace jxltran
