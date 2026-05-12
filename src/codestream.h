// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// In-memory model of a bare JXL codestream: one image header and a sequence of
// frames. Each frame keeps a parsed FrameHeader plus exact ranges into the
// source codestream.
//
// Frame bit layout: [frame_header][TOC][body]. For rewrite we emit a new
// frame_header, copy the TOC verbatim for now (later: re-encode a semantically
// equivalent TOC -- U32 size codewords are unique; only the permutation stream
// may have multiple encodings -- and permute body bytes to match). Then copy the
// body. The body is byte-aligned in the codestream (the TOC ends on a byte
// boundary); group sections are whole bytes.

#ifndef TOOLS_JXLTRAN_CODESTREAM_H_
#define TOOLS_JXLTRAN_CODESTREAM_H_

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "frame_header.h"
#include "image_header.h"

namespace jxltran {

struct FramedUnit {
  FrameHeader frame;
  // Logical header for rewrite purposes is frame_header + TOC; the body is
  // everything after the TOC (byte-aligned in the codestream).
  // Bit offset from the start of this frame where the TOC begins (after the
  // frame header bits).
  size_t toc_start_bit = 0;
  // Length in bits of the TOC only (verbatim from input when writing).
  size_t toc_bit_length = 0;
  // Length in bits of compressed frame sections after the TOC; always a
  // multiple of 8 (whole bytes).
  size_t body_bit_length = 0;
  // Total byte length of this frame in the source codestream (header through
  // last frame data byte). Used to copy LF / ReferenceOnly frames verbatim.
  size_t full_frame_byte_len = 0;
  // Byte offset from the start of the codestream where this frame begins
  // (used as |original_data| base for WriteFrameHeader verbatim RF/ICC bits).
  size_t original_frame_byte_offset = 0;

  // TOC decoded for re-encode when frame header / TOC bit phase changes.
  // Empty |toc_perm| means no permutation. Used to re-encode the TOC only when
  // a non-empty permutation is present, WriteCodestream fails with an error
  // (TOC permutation re-encode is not implemented in the standalone tool).
  std::vector<uint32_t> toc_decoded_sizes;
  std::vector<uint32_t> toc_perm;  // coeff_order_t values; empty if none

  // Photon-noise DC-global splice (see ApplyPhotonNoiseIso). Patch point is an
  // absolute codestream bit index (from first codestream byte = bit 0).
  bool photon_noise_edit = false;
  int32_t photon_noise_delta_bytes = 0;  // -10, 0, or +10
  size_t photon_noise_patch_abs_bit = 0;
  std::array<uint8_t, 10> photon_noise_new_bytes = {};
  bool photon_noise_toc_reencode = false;
};

struct ParsedCodestream {
  ImageHeader image;
  std::vector<FramedUnit> frames;
};

// Parses a complete codestream (0xFF0A …) into |out|. Returns false on error.
bool ReadCodestream(const uint8_t* data, size_t size, ParsedCodestream* out);

// Serializes |cs| to |out|. |original| must point at the same codestream bytes
// used when |cs| was produced by ReadCodestream (ICC entropy bits and frame RF
// ranges are copied verbatim relative to per-frame offsets).
bool WriteCodestream(const ParsedCodestream& cs, const uint8_t* original,
                     std::vector<uint8_t>* out);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_CODESTREAM_H_
