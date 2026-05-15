// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// JXL frame header structs and parser/writer for jxltran.
// Parses the frame header and TOC for each frame so that cropping can be
// implemented as a metadata-only bitstream modification.
//
// Parsed frame headers keep the loop filter and frame-level extensions; when
// |rf_reencode| is false they are copied verbatim from the source codestream
// on write (unless the whole frame header packs as one all-default bit).
//
// References:
//   JXL spec appendix D "Frame header" (40-frame-header.adoc)
//   JXL spec appendix G "Restoration filters" (70-filters.adoc)

#ifndef TOOLS_JXLTRAN_FRAME_HEADER_H_
#define TOOLS_JXLTRAN_FRAME_HEADER_H_

#include <cstddef>
#include <cstdint>
#include <vector>

#include "bits.h"
#include "image_header.h"
#include "restoration_filter.h"

namespace jxltran {

// ── Frame types and encoding ─────────────────────────────────────────────────

static constexpr uint32_t kFrameTypeRegular = 0;
static constexpr uint32_t kFrameTypeLF = 1;
static constexpr uint32_t kFrameTypeReferenceOnly = 2;
static constexpr uint32_t kFrameTypeSkipProgressive = 3;

static constexpr uint32_t kFrameEncVarDCT = 0;
static constexpr uint32_t kFrameEncModular = 1;

static constexpr uint64_t kFlagUseLfFrame = 32;

// lib/jxl/frame_header.h FrameHeader::Flags (subset used by jxltran).
static constexpr uint64_t kFrameFlagNoise = 1;
static constexpr uint64_t kFrameFlagPatches = 2;
static constexpr uint64_t kFrameFlagSplines = 16;

// ── Sub-structs
// ───────────────────────────────────────────────────────────────

struct FramePasses {
  uint32_t num_passes = 1;
  uint32_t num_ds = 0;
  uint32_t shift[8] = {};
  uint32_t downsample[8] = {};
  uint32_t last_pass[8] = {};
};

struct FrameBlendingInfo {
  uint32_t mode = 0;           // BlendMode: 0=kReplace, 1=kAdd, 2=kBlend, ...
  uint32_t alpha_channel = 0;  // present when extra && (kBlend||kMulAdd)
  bool clamp = false;          // present when extra && (kBlend||kMulAdd||kMul)
  uint32_t source = 0;         // present when !resets_canvas
};

// ── FrameHeader
// ───────────────────────────────────────────────────────────────

struct FrameHeader {
  // Legacy field: jxltran always keeps an expanded header in memory after read.
  // On write, whether the header packs as a single all-default bit is computed
  // from field values (see ComputeFrameHeaderPacksAsOneDefaultBit in
  // frame_header.cc), not from this member.
  bool all_default = false;

  // Fields serialized when the header is not packed as one all-default bit:
  uint32_t frame_type = kFrameTypeRegular;
  uint32_t encoding = kFrameEncVarDCT;
  uint64_t flags = 0;

  bool do_YCbCr = false;             // only if !xyb_encoded
  uint32_t jpeg_upsampling[3] = {};  // only if do_YCbCr && !kUseLfFrame

  uint32_t upsampling = 1;  // only if !kUseLfFrame
  std::vector<uint32_t>
      ec_upsampling;  // num_extra entries, only if !kUseLfFrame

  uint32_t group_size_shift = 1;  // only if encoding == kModular
  uint32_t x_qm_scale = 3;        // only if xyb_encoded && kVarDCT
  uint32_t b_qm_scale = 2;        // only if xyb_encoded && kVarDCT

  FramePasses passes;  // only if frame_type != kReferenceOnly

  uint32_t lf_level = 0;  // 1-4 for kLFFrame; from 1+u(2) in the bitstream

  bool have_crop = false;  // only if !all_default && frame_type != kLFFrame
  uint32_t
      ux0 = 0,
      uy0 = 0;  // packed-signed offsets; only if have_crop && !kReferenceOnly
  uint32_t crop_width = 0;   // only if have_crop
  uint32_t crop_height = 0;  // only if have_crop

  // BlendingInfo: only if !all_default && normal_frame
  FrameBlendingInfo blending_info;
  std::vector<FrameBlendingInfo> ec_blending_info;  // num_extra entries

  uint32_t duration =
      0;  // only if !all_default && normal_frame && have_animation
  uint32_t timecode =
      0;  // only if !all_default && normal_frame && have_timecodes

  bool is_last =
      true;  // only if !all_default && normal_frame; default=!frame_type
  uint32_t save_as_reference =
      0;  // only if !all_default && frame_type!=kLFFrame && !is_last

  bool save_before_ct = false;  // only in specific conditions (see spec)
  bool save_before_ct_present =
      false;  // whether save_before_ct was in the bitstream

  std::vector<uint8_t> name;  // present when !all_default (but length may be 0)

  // RestorationFilter (loop filter bundle) + frame-level Extensions after it.
  // |rf_start_bit|..|frame_extensions_start_bit| is the loop filter (including
  // its own extensions); |frame_extensions_start_bit|..|rf_ext_end_bit| is
  // the FrameHeader extensions (U64 mask + payload, see spec).
  size_t rf_start_bit = 0;
  size_t frame_extensions_start_bit = 0;
  size_t rf_ext_end_bit = 0;
  // Parsed RF when |rf_reencode| or when read from an expanded header; when
  // the loop filter is the implicit default, |restoration.all_default| is true.
  ParsedRestorationFilter restoration;
  bool rf_reencode = false;

  // First U64 of FrameHeader extensions (after the loop filter bundle). Used
  // to decide whether the whole header can pack as one all-default bit (mask
  // must be 0).
  uint64_t frame_extensions_mask = 0;

  // First bit of have_crop in the source frame (!all_default, not LF). Used
  // to verbatim-copy the unchanged prefix before have_crop when rewriting.
  size_t crop_bundle_start_bit = 0;
};

// ── API
// ───────────────────────────────────────────────────────────────────────

// Parses a JXL frame header from |br| (which must be byte-aligned on entry).
// |meta| provides image-level context (xyb_encoded, num_extra, etc.).
// |canvas_w|, |canvas_h|: current canvas dimensions (for resets_canvas logic).
// |original_data| is the base of the buffer that |br| was constructed from
// (used to record the verbatim RestorationFilter/Extensions bit range).
// On return, |br| is at the next byte boundary (after the frame header).
// Returns false on parse error.
bool ReadFrameHeader(BitReader& br, const ImageMetadata& meta,
                     uint32_t canvas_w, uint32_t canvas_h, FrameHeader* fh,
                     const uint8_t* original_data);

// Writes the frame header to |bw| (no extra byte padding after the header).
// Emits a single all-default true bit iff the header matches implicit defaults
// (including restoration.all_default, no rf re-encode, and no frame
// extensions). Otherwise writes the full header; loop filter + frame
// extensions are copied verbatim from |original_data| unless |rf_reencode|.
// |canvas_w|/|canvas_h| are the image header canvas dimensions; they must
// match ReadFrameHeader for BlendingInfo (partial frame / source bits) and
// save_before_ct layout, matching libjxl's use of metadata canvas size.
bool WriteFrameHeader(BitWriter& bw, const FrameHeader& fh,
                      const ImageMetadata& meta, const uint8_t* original_data,
                      uint32_t canvas_w, uint32_t canvas_h);

// Same bits as the tail of WriteFrameHeader from have_crop through RF +
// extensions. Caller must emit all bits before have_crop (identical to
// source) separately.
bool WriteFrameHeaderFromHaveCrop(BitWriter& bw, const FrameHeader& fh,
                                  const ImageMetadata& meta,
                                  const uint8_t* original_data,
                                  uint32_t canvas_w, uint32_t canvas_h);

// Parsed frame dimensions used for TOC entry counts and layout labels
// (matches libjxl FrameDimensions + NumTocEntries / AcGroupIndex).
struct FrameTocMetrics {
  size_t num_toc_entries = 0;
  // True when num_groups==1 && num_passes==1 (single combined TOC section).
  bool all_in_one_section = false;
  size_t num_groups = 0;
  size_t num_dc_groups = 0;
  size_t num_passes = 1;
  size_t group_dim = 0;
  size_t xsize = 0;
  size_t ysize = 0;
  size_t xsize_blocks = 0;
  size_t ysize_blocks = 0;
  size_t xsize_groups = 0;
  size_t ysize_groups = 0;
  size_t xsize_dc_groups = 0;
  size_t ysize_dc_groups = 0;
};

// Fills |out| from frame header + canvas (same geometry as libjxl ToFrameDimensions).
void ComputeFrameTocMetrics(const FrameHeader& fh, const ImageMetadata& meta,
                            uint32_t canvas_w, uint32_t canvas_h,
                            FrameTocMetrics* out);

// True when blending_info serialization includes the `source` field (partial
// frame or non-replace mode), matching ReadBlendingInfo / WriteBlendingInfo.
bool FrameNeedsBlendingSourceField(const FrameHeader& fh, uint32_t canvas_w,
                                   uint32_t canvas_h);

// Parses the TOC that follows the frame header (toc.cc).
// |br| must have consumed the frame header from the start of the frame span.
// |frame_base|/|frame_len| are reserved for future bounds checks. On success,
// |br| is advanced past the TOC.
struct FramedUnit;

bool ReadFrameTOC(BitReader& br, const FrameHeader& fh,
                  const ImageMetadata& meta, uint32_t canvas_w, uint32_t canvas_h,
                  const uint8_t* frame_base, size_t frame_len,
                  uint64_t* frame_data_bytes, FramedUnit* fu);

// Packs a signed int32 into an unsigned uint32 (JXL signed-int packing).
//   PackSigned(s)  = s >= 0 ? 2*s : -2*s - 1
//   UnpackSigned(u)= u even ? u/2 : -(u/2)-1
inline uint32_t PackSigned(int32_t s) {
  return s >= 0 ? static_cast<uint32_t>(s) * 2
                : static_cast<uint32_t>(-s) * 2 - 1;
}
inline int32_t UnpackSigned(uint32_t u) {
  return (u & 1) ? -static_cast<int32_t>((u + 1) / 2)
                 : static_cast<int32_t>(u / 2);
}

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_FRAME_HEADER_H_
