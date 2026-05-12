// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "codestream.h"

#include <array>
#include <cstdio>

#include "printf_macros.h"
#include "toc.h"
#include "trace.h"

namespace jxltran {

namespace {

void AppendNoise80Bits(BitWriter* bw, const std::array<uint8_t, 10>& nb) {
  for (size_t i = 0; i < 80; ++i) {
    bw->WriteBits(1, (nb[i >> 3] >> (i & 7)) & 1);
  }
}

}  // namespace

bool ReadCodestream(const uint8_t* data, size_t size, ParsedCodestream* out) {
  out->frames.clear();
  BitReader br(data, size);
  TraceSetReadBase(0);
  if (TraceIsOn()) {
    TraceMarkerAt(0, "read.codestream", "size_bytes=%" PRIuS "", size);
  }
  if (!ReadImageHeader(br, &out->image)) return false;
  if (!br.ok()) return false;
  if ((br.pos() & 7u) != 0) return false;
  size_t pos = br.pos() / 8;
  if (TraceIsOn()) {
    TraceMarkerAt(br.pos(), "read.codestream.after_image_header",
                  "next_frame_byte=%" PRIuS "", pos);
  }
  const uint32_t canvas_w = out->image.size.width;
  const uint32_t canvas_h = out->image.size.height;

  size_t frame_index = 0;
  while (pos < size) {
    FramedUnit fu;
    fu.original_frame_byte_offset = pos;
    const uint8_t* frame_base = data + pos;
    const size_t frame_avail = size - pos;
    TraceSetReadBase(pos * 8);
    if (TraceIsOn()) {
      TraceMarkerAt(pos * 8, "read.codestream.frame",
                    "index=%" PRIuS " byte_off=%" PRIuS "", frame_index, pos);
    }
    BitReader fbr(frame_base, frame_avail);
    if (!ReadFrameHeader(fbr, out->image.metadata, canvas_w, canvas_h,
                         &fu.frame, frame_base)) {
      fprintf(stderr,
              "jxltran: failed to parse frame header at codestream %" PRIuS
              "+%u\n",
              (pos * 8 + fbr.pos()) / 8u,
              static_cast<unsigned>((pos * 8 + fbr.pos()) & 7u));
      return false;
    }
    fu.toc_start_bit = fbr.pos();
    uint64_t frame_data_bytes = 0;
    if (!ReadFrameTOC(fbr, fu.frame, out->image.metadata, canvas_w, canvas_h,
                      frame_base, frame_avail, &frame_data_bytes, &fu)) {
      const size_t abs_bit = pos * 8 + fbr.pos();
      fprintf(stderr,
              "jxltran: failed to parse TOC at codestream %" PRIuS
              "+%u (frame rel bits=%" PRIuS ")\n",
              abs_bit / 8u, static_cast<unsigned>(abs_bit & 7u), fbr.pos());
      return false;
    }
    const size_t body_start_bit = fbr.pos();
    for (uint64_t i = 0; i < frame_data_bytes; ++i) {
      (void)fbr.ReadBits(8);
    }
    if (!fbr.ok()) {
      fprintf(stderr,
              "jxltran: frame data overruns codestream at codestream %" PRIuS
              "+%u\n",
              (pos * 8 + fbr.pos()) / 8u,
              static_cast<unsigned>((pos * 8 + fbr.pos()) & 7u));
      return false;
    }
    if ((fbr.pos() & 7u) != 0) {
      fprintf(stderr,
              "jxltran: frame at codestream %" PRIuS "+%u does not end on a "
              "byte boundary\n",
              (pos * 8 + fbr.pos()) / 8u,
              static_cast<unsigned>((pos * 8 + fbr.pos()) & 7u));
      return false;
    }
    fu.body_bit_length = fbr.pos() - body_start_bit;
    if ((body_start_bit & 7u) != 0) {
      fprintf(stderr,
              "jxltran: frame at codestream %" PRIuS "+%u: body does not start on "
              "a byte boundary after TOC\n",
              (pos * 8 + body_start_bit) / 8u,
              static_cast<unsigned>((pos * 8 + body_start_bit) & 7u));
      return false;
    }
    if ((fu.body_bit_length & 7u) != 0) {
      fprintf(stderr,
              "jxltran: frame at codestream %" PRIuS "+%u: body length is not a "
              "whole number of bytes\n",
              (pos * 8 + fbr.pos()) / 8u,
              static_cast<unsigned>((pos * 8 + fbr.pos()) & 7u));
      return false;
    }
    fu.full_frame_byte_len = fbr.pos() / 8;
    if (fu.full_frame_byte_len == 0) {
      fprintf(stderr,
              "jxltran: frame at codestream %" PRIuS "+%u has zero length "
              "(corrupt codestream)\n",
              pos, 0u);
      return false;
    }
    if (pos + fu.full_frame_byte_len > size ||
        pos + fu.full_frame_byte_len < pos) {
      fprintf(stderr,
              "jxltran: frame size overruns codestream at byte %" PRIuS "\n",
              pos);
      return false;
    }
    pos += fu.full_frame_byte_len;
    out->frames.push_back(std::move(fu));
    if (TraceIsOn()) {
      TraceMarkerAt(pos * 8, "read.codestream.frame_done",
                    "index=%" PRIuS " full_frame_bytes=%" PRIuS "", frame_index,
                    out->frames.back().full_frame_byte_len);
    }
    ++frame_index;
    if (out->frames.back().frame.is_last) break;
  }
  return br.ok();
}

bool WriteCodestream(const ParsedCodestream& cs, const uint8_t* original,
                     std::vector<uint8_t>* out) {
  if (original == nullptr) {
    fprintf(stderr,
            "jxltran: WriteCodestream requires original codestream bytes\n");
    return false;
  }
  BitWriter bw;
  if (TraceIsOn()) {
    TraceMarkerAt(bw.bit_pos(), "write.codestream", "begin");
  }
  WriteImageHeader(bw, cs.image, original);
  const uint32_t meta_canvas_w = cs.image.size.width;
  const uint32_t meta_canvas_h = cs.image.size.height;
  size_t wframe = 0;
  for (const FramedUnit& fu : cs.frames) {
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) {
      if (TraceIsOn()) {
        TraceMarkerAt(
            bw.bit_pos(), "write.codestream.frame_verbatim",
            "index=%" PRIuS " type=LF_or_RefOnly bytes=%" PRIuS "", wframe,
            fu.full_frame_byte_len);
      }
      bw.ZeroPadToByte();
      bw.AppendRawBytes(original + fu.original_frame_byte_offset,
                        fu.full_frame_byte_len);
      ++wframe;
      continue;
    }
    const uint8_t* frame_base = original + fu.original_frame_byte_offset;
    const size_t frame_bit_base = fu.original_frame_byte_offset * 8;
    const size_t toc_rel = fu.toc_start_bit;
    // Body always follows the TOC in the *source* codestream at this bit offset
    // (independent of whether we re-encoded the TOC to a different bit length).
    const size_t body_src_abs_bit =
        frame_bit_base + toc_rel + fu.toc_bit_length;
    if (TraceIsOn()) {
      TraceMarkerAt(bw.bit_pos(), "write.codestream.frame",
                    "index=%" PRIuS " src_byte=%" PRIuS " toc_bits=%" PRIuS
                    " body_bits=%" PRIuS "",
                    wframe, fu.original_frame_byte_offset, fu.toc_bit_length,
                    fu.body_bit_length);
    }
    // [frame_header] from |fu.frame|, then verbatim TOC, then verbatim body.
    // The frame header may change length (e.g. crop); TOC is unchanged for now.
    // The body is whole bytes; use AppendRawBytes when the writer is aligned.
    if (!WriteFrameHeader(bw, fu.frame, cs.image.metadata, frame_base,
                          meta_canvas_w, meta_canvas_h)) {
      fprintf(stderr, "jxltran: failed to write frame header\n");
      return false;
    }
    const size_t phase_out = bw.bit_pos() % 8u;
    const size_t phase_in = (frame_bit_base + toc_rel) % 8u;
    if (TraceIsOn()) {
      TraceWriteField(bw.bit_pos(), "write.frame.splice_toc_bits",
                      "src_start_bit=%" PRIuS " count=%" PRIuS " reencode=%d",
                      frame_bit_base + toc_rel, fu.toc_bit_length,
                      (phase_out != phase_in) ? 1 : 0);
    }
    const bool photon_toc =
        fu.photon_noise_edit && fu.photon_noise_toc_reencode;
    if (phase_out == phase_in && !photon_toc) {
      bw.AppendBitsRange(original, frame_bit_base + toc_rel, fu.toc_bit_length);
    } else {
      const size_t phase_for_toc = bw.bit_pos() % 8u;
      if (!WriteDecodedToc(bw, phase_for_toc, fu.toc_perm, fu.toc_decoded_sizes)) {
        fprintf(stderr,
                "jxltran: failed to re-encode TOC (header/TOC bit phase "
                "changed)\n");
        return false;
      }
    }
    if (TraceIsOn()) {
      TraceWriteField(bw.bit_pos(), "write.frame.splice_body",
                      "src_start_bit=%" PRIuS " bits=%" PRIuS " aligned=%d",
                      body_src_abs_bit, fu.body_bit_length,
                      (bw.bit_pos() & 7u) == 0 && (body_src_abs_bit & 7u) == 0
                          ? 1
                          : 0);
    }
    if (!fu.photon_noise_edit) {
      if ((bw.bit_pos() & 7u) == 0u && (body_src_abs_bit & 7u) == 0u) {
        bw.AppendRawBytes(original + body_src_abs_bit / 8,
                            fu.body_bit_length / 8);
      } else {
        bw.AppendBitsRange(original, body_src_abs_bit, fu.body_bit_length);
      }
    } else {
      const int64_t delta_bits =
          static_cast<int64_t>(fu.photon_noise_delta_bytes) * 8;
      const size_t old_body_bits =
          static_cast<size_t>(static_cast<int64_t>(fu.body_bit_length) -
                              delta_bits);
      const size_t body_end_orig = body_src_abs_bit + old_body_bits;
      const size_t p = fu.photon_noise_patch_abs_bit;
      if ((p & 7u) != 0 || p < body_src_abs_bit || p > body_end_orig) {
        fprintf(stderr, "jxltran: photon noise splice: bad patch offset\n");
        return false;
      }
      if (fu.photon_noise_delta_bytes < 0) {
        if (p + 80 > body_end_orig) {
          fprintf(stderr, "jxltran: photon noise splice: noise past body end\n");
          return false;
        }
        bw.AppendBitsRange(original, body_src_abs_bit, p - body_src_abs_bit);
        bw.AppendBitsRange(original, p + 80, body_end_orig - (p + 80));
      } else if (fu.photon_noise_delta_bytes == 0) {
        if (p + 80 > body_end_orig) {
          fprintf(stderr, "jxltran: photon noise splice: noise past body end\n");
          return false;
        }
        bw.AppendBitsRange(original, body_src_abs_bit, p - body_src_abs_bit);
        AppendNoise80Bits(&bw, fu.photon_noise_new_bytes);
        bw.AppendBitsRange(original, p + 80, body_end_orig - (p + 80));
      } else {
        bw.AppendBitsRange(original, body_src_abs_bit, p - body_src_abs_bit);
        AppendNoise80Bits(&bw, fu.photon_noise_new_bytes);
        bw.AppendBitsRange(original, p, body_end_orig - p);
      }
    }
    ++wframe;
  }
  *out = bw.TakeBytes();
  return true;
}

}  // namespace jxltran
