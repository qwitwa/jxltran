// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "toc_layout.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <map>
#include <string>

#include "box.h"
#include "codestream.h"
#include <cstring>
#include "frame_header.h"
#include "printf_macros.h"

namespace jxltran {

namespace {

static constexpr size_t kBlockDim = 8;

static size_t AcGroupIndex(size_t pass, size_t group, size_t num_groups,
                           size_t num_dc_groups) {
  return 2 + num_dc_groups + pass * num_groups + group;
}

static uint32_t LoadBe32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

static void JsonEscString(FILE* out, const char* s, size_t len) {
  fputc('"', out);
  for (size_t i = 0; i < len; ++i) {
    unsigned char c = static_cast<unsigned char>(s[i]);
    if (c == '"' || c == '\\') {
      fputc('\\', out);
      fputc(static_cast<char>(c), out);
    } else if (c < 32) {
      fprintf(out, "\\u%04x", c);
    } else {
      fputc(static_cast<char>(c), out);
    }
  }
  fputc('"', out);
}

}  // namespace

std::string TocLogicalSectionLabel(const FrameTocMetrics& m,
                                   const FrameHeader& fh, size_t logical) {
  if (m.num_toc_entries != 0 && logical >= m.num_toc_entries) {
    return "invalid_logical";
  }
  if (m.all_in_one_section) {
    return "combined";
  }
  const size_t ndc = m.num_dc_groups;
  if (logical == 0) {
    return "lf_global";
  }
  if (logical >= 1 && logical <= ndc) {
    const size_t dci = logical - 1;
    const size_t gx = dci % m.xsize_dc_groups;
    const size_t gy = dci / m.xsize_dc_groups;
    const size_t x0b = gx * m.group_dim;
    const size_t y0b = gy * m.group_dim;
    const size_t wb =
        std::min(m.group_dim, m.xsize_blocks > x0b ? m.xsize_blocks - x0b : 0u);
    const size_t hb =
        std::min(m.group_dim, m.ysize_blocks > y0b ? m.ysize_blocks - y0b : 0u);
    const size_t x0p = x0b * kBlockDim;
    const size_t y0p = y0b * kBlockDim;
    const size_t wp = wb * kBlockDim;
    const size_t hp = hb * kBlockDim;
    char buf[160];
    snprintf(buf, sizeof(buf),
             "lf_group %" PRIuS " (%" PRIuS "x%" PRIuS "+%" PRIuS "+%" PRIuS ")",
             dci, wp, hp, x0p, y0p);
    return std::string(buf);
  }
  const size_t global_ac = ndc + 1;
  if (logical == global_ac) {
    return fh.encoding == kFrameEncVarDCT ? "hf_global" : "pass_global";
  }
  const size_t ac0 = AcGroupIndex(0, 0, m.num_groups, ndc);
  const size_t rel = logical - ac0;
  const size_t ng = m.num_groups;
  const size_t pass = rel / ng;
  const size_t grp = rel % ng;
  const size_t gx = grp % m.xsize_groups;
  const size_t gy = grp / m.xsize_groups;
  const size_t x0 = gx * m.group_dim;
  const size_t y0 = gy * m.group_dim;
  const size_t w =
      std::min(m.group_dim, m.xsize > x0 ? m.xsize - x0 : 0u);
  const size_t h =
      std::min(m.group_dim, m.ysize > y0 ? m.ysize - y0 : 0u);
  char buf[200];
  snprintf(buf, sizeof(buf),
           "hf_pass %" PRIuS " group %" PRIuS " (%" PRIuS "x%" PRIuS "+%" PRIuS
           "+%" PRIuS ")",
           pass, grp, w, h, x0, y0);
  return std::string(buf);
}

void PrintCodestreamTocLayoutVerbose(FILE* out, const ParsedCodestream& cs) {
  const uint32_t cw = cs.image.size.width;
  const uint32_t ch = cs.image.size.height;

  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    const FramedUnit& fu = cs.frames[fi];
    FrameTocMetrics tm;
    ComputeFrameTocMetrics(fu.frame, cs.image.metadata, cw, ch, &tm);

    fprintf(out, "toc_verbose: frame %" PRIuS " toc_entries=%" PRIuS
                " permutation=%s num_groups=%" PRIuS " num_dc_groups=%" PRIuS
                " num_passes=%" PRIuS " group_dim=%" PRIuS "\n",
            fi, tm.num_toc_entries,
            (!fu.toc_perm.empty() ? "yes" : "no"), tm.num_groups,
            tm.num_dc_groups, tm.num_passes, tm.group_dim);

    if (fu.toc_decoded_sizes.empty()) {
      fprintf(out, "toc_verbose: frame %" PRIuS ": (no decoded TOC sizes)\n",
              fi);
      continue;
    }
    if (fu.toc_decoded_sizes.size() != tm.num_toc_entries) {
      fprintf(out,
              "toc_verbose: frame %" PRIuS ": WARNING sizes.size=%" PRIuS
              " != expected toc_entries=%" PRIuS "\n",
              fi, fu.toc_decoded_sizes.size(), tm.num_toc_entries);
    }
    if (!fu.toc_perm.empty() && fu.toc_perm.size() != fu.toc_decoded_sizes.size()) {
      fprintf(out,
              "toc_verbose: frame %" PRIuS ": WARNING perm.size=%" PRIuS
              " != sizes.size=%" PRIuS "\n",
              fi, fu.toc_perm.size(), fu.toc_decoded_sizes.size());
    }

    const size_t n = std::min(fu.toc_decoded_sizes.size(), tm.num_toc_entries);
    const size_t body0_byte =
        fu.original_frame_byte_offset +
        (fu.toc_start_bit + fu.toc_bit_length) / 8u;
    size_t body_cursor = 0;
    for (size_t stream = 0; stream < n; ++stream) {
      const uint32_t sz = fu.toc_decoded_sizes[stream];
      const size_t logical =
          fu.toc_perm.empty() ? stream
                              : (stream < fu.toc_perm.size() ? fu.toc_perm[stream]
                                                              : stream);
      const size_t abs_byte = body0_byte + body_cursor;
      const std::string label = TocLogicalSectionLabel(tm, fu.frame, logical);
      fprintf(out,
              "toc_verbose: frame %" PRIuS " offset %" PRIuS
              " stream[%" PRIuS "] logical[%" PRIuS "] size %" PRIu32 " %s\n",
              fi, abs_byte, stream, logical, sz, label.c_str());
      body_cursor += sz;
    }
  }
}

void PrintBitstreamStructureJson(FILE* out, const uint8_t* file_data,
                                 size_t file_size,
                                 const std::vector<JxlBox>& boxes,
                                 bool is_container_file,
                                 const ParsedCodestream& cs,
                                 size_t codestream_byte_length) {
  (void)file_data;
  fputc('{', out);
  fprintf(out, "\"version\":1,\"file_size\":%" PRIuS ",\"container_file\":%s,",
          file_size, is_container_file ? "true" : "false");

  fputs("\"boxes\":[", out);
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (i) fputc(',', out);
    const JxlBox& b = boxes[i];
    fputc('{', out);
    fprintf(out, "\"index\":%" PRIuS ",\"type\":", i);
    JsonEscString(out, b.type, 4);
    fprintf(out, ",\"byte_start\":%" PRIuS ",\"byte_end_exclusive\":%" PRIuS,
            b.file_offset, b.file_offset + b.file_total_bytes);
    fprintf(out, ",\"header_bytes\":%u",
            static_cast<unsigned>(b.header_size));
    fprintf(out, ",\"payload_byte_start\":%" PRIuS,
            b.file_offset + b.header_size);
    fprintf(out, ",\"payload_byte_end_exclusive\":%" PRIuS,
            b.file_offset + b.file_total_bytes);
    if (memcmp(b.type, "brob", 4) == 0 && b.data.size() >= 4) {
      fputs(",\"brob_inner_type\":", out);
      JsonEscString(out, reinterpret_cast<const char*>(b.data.data()), 4);
    }
    if (memcmp(b.type, "jxlp", 4) == 0 && b.data.size() >= 4) {
      const uint32_t word = LoadBe32(b.data.data());
      fprintf(out, ",\"jxlp_counter\":%" PRIu32 ",\"jxlp_is_last\":%s",
              word & 0x7FFFFFFFu, (word & 0x80000000u) ? "true" : "false");
    }
    if (memcmp(b.type, "ftyp", 4) == 0 && b.data.size() >= 8) {
      fputs(",\"ftyp_major_brand\":", out);
      JsonEscString(out, reinterpret_cast<const char*>(b.data.data()), 4);
      const uint32_t minor = LoadBe32(b.data.data() + 4);
      fprintf(out, ",\"ftyp_minor_version\":%" PRIu32, minor);
    }
    fputc('}', out);
  }
  fputs("],", out);

  fputs("\"codestream_fragments\":[", out);
  bool has_jxlc = false;
  bool has_jxlp = false;
  for (const auto& b : boxes) {
    if (memcmp(b.type, "jxlc", 4) == 0) has_jxlc = true;
    if (memcmp(b.type, "jxlp", 4) == 0) has_jxlp = true;
  }
  size_t frag_ix = 0;
  size_t cs_acc = 0;
  if (has_jxlc && !has_jxlp) {
    for (const auto& b : boxes) {
      if (memcmp(b.type, "jxlc", 4) != 0) continue;
      if (frag_ix++) fputc(',', out);
      const size_t len = b.data.size();
      const size_t pay0 = b.file_offset + b.header_size;
      const size_t pay1 = b.file_offset + b.file_total_bytes;
      fprintf(out,
              "{\"kind\":\"jxlc\",\"codestream_byte_start\":%" PRIuS
              ",\"codestream_byte_end\":%" PRIuS
              ",\"file_payload_byte_start\":%" PRIuS
              ",\"file_payload_byte_end_exclusive\":%" PRIuS "}",
              cs_acc, cs_acc + len, pay0, pay1);
      cs_acc += len;
    }
  } else if (has_jxlp && !has_jxlc) {
    std::map<uint32_t, const JxlBox*> by_index;
    for (const auto& b : boxes) {
      if (memcmp(b.type, "jxlp", 4) != 0) continue;
      if (b.data.size() < 4) continue;
      const uint32_t word = LoadBe32(b.data.data());
      const uint32_t idx = word & 0x7FFFFFFFu;
      by_index[idx] = &b;
    }
    for (const auto& kv : by_index) {
      const JxlBox& b = *kv.second;
      if (frag_ix++) fputc(',', out);
      const size_t payload_len =
          b.data.size() >= 4 ? b.data.size() - 4 : 0;
      const size_t pay0 = b.file_offset + b.header_size + 4;
      const size_t pay1 = b.file_offset + b.file_total_bytes;
      const uint32_t word = LoadBe32(b.data.data());
      fprintf(out,
              "{\"kind\":\"jxlp\",\"jxlp_counter\":%" PRIu32
              ",\"jxlp_is_last\":%s,"
              "\"codestream_byte_start\":%" PRIuS
              ",\"codestream_byte_end\":%" PRIuS
              ",\"file_payload_byte_start\":%" PRIuS
              ",\"file_payload_byte_end_exclusive\":%" PRIuS "}",
              kv.first, (word & 0x80000000u) ? "true" : "false", cs_acc,
              cs_acc + payload_len, pay0, pay1);
      cs_acc += payload_len;
    }
  } else if (!is_container_file) {
    fprintf(out,
            "{\"kind\":\"bare\",\"codestream_byte_start\":0,"
            "\"codestream_byte_end\":%" PRIuS
            ",\"file_payload_byte_start\":0,"
            "\"file_payload_byte_end_exclusive\":%" PRIuS "}",
            codestream_byte_length, file_size);
  }
  fputs("],", out);

  fputs("\"codestream\":{", out);
  fprintf(out, "\"byte_length\":%" PRIuS ",\"width\":%" PRIu32
              ",\"height\":%" PRIu32 ",",
          codestream_byte_length, cs.image.size.width, cs.image.size.height);
  size_t ih_end = 0;
  if (!cs.frames.empty()) {
    ih_end = cs.frames[0].original_frame_byte_offset;
  } else {
    ih_end = codestream_byte_length;
  }
  fprintf(out, "\"image_header\":{\"byte_start\":0,\"byte_end_exclusive\":%" PRIuS
              "},",
          ih_end);

  fputs("\"frames\":[", out);
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (fi) fputc(',', out);
    const FramedUnit& fu = cs.frames[fi];
    const size_t fb = fu.original_frame_byte_offset;
    const size_t frame_end = fb + fu.full_frame_byte_len;
    const size_t fh_bit0 = fb * 8;
    const size_t toc_bit0 = fh_bit0 + fu.toc_start_bit;
    const size_t body_bit0 = toc_bit0 + fu.toc_bit_length;
    const size_t body_bit1 = body_bit0 + fu.body_bit_length;
    FrameTocMetrics tm{};
    ComputeFrameTocMetrics(fu.frame, cs.image.metadata, cs.image.size.width,
                          cs.image.size.height, &tm);
    fputc('{', out);
    fprintf(out, "\"index\":%" PRIuS ",", fi);
    if (!fu.frame.name.empty()) {
      fputs("\"name_hex\":\"", out);
      for (uint8_t c : fu.frame.name) fprintf(out, "%02x", c);
      fputs("\",", out);
    }
    fprintf(out, "\"byte_start\":%" PRIuS ",\"byte_end_exclusive\":%" PRIuS ",",
            fb, frame_end);
    fprintf(out, "\"width\":%" PRIuS ",\"height\":%" PRIuS ",",
            tm.xsize, tm.ysize);
    fprintf(out, "\"frame_header\":{\"bit_start\":%" PRIuS ",\"bit_end\":%" PRIuS,
            fh_bit0, toc_bit0);
    fprintf(out, ",\"byte_start\":%" PRIuS ",\"byte_end_exclusive\":%" PRIuS "},",
            fb, (toc_bit0 + 7u) / 8u);
    fprintf(out, "\"toc\":{\"bit_start\":%" PRIuS ",\"bit_end\":%" PRIuS,
            toc_bit0, body_bit0);
    fprintf(out, ",\"byte_start\":%" PRIuS ",\"byte_end_exclusive\":%" PRIuS "},",
            toc_bit0 / 8u, (body_bit0 + 7u) / 8u);
    fprintf(out, "\"body\":{\"bit_start\":%" PRIuS ",\"bit_end\":%" PRIuS,
            body_bit0, body_bit1);
    fprintf(out, ",\"byte_start\":%" PRIuS ",\"byte_end_exclusive\":%" PRIuS "}",
            body_bit0 / 8u, (body_bit1 + 7u) / 8u);

    fputs(",\"toc_sections\":[", out);
    if (!fu.toc_decoded_sizes.empty()) {
      const size_t n =
          std::min(fu.toc_decoded_sizes.size(), tm.num_toc_entries);
      const size_t body0_byte = fu.original_frame_byte_offset +
                                (fu.toc_start_bit + fu.toc_bit_length) / 8u;
      size_t body_cursor = 0;
      for (size_t stream = 0; stream < n; ++stream) {
        if (stream) fputc(',', out);
        const uint32_t sz = fu.toc_decoded_sizes[stream];
        const size_t logical =
            fu.toc_perm.empty()
                ? stream
                : (stream < fu.toc_perm.size() ? fu.toc_perm[stream] : stream);
        const size_t abs_byte = body0_byte + body_cursor;
        const std::string label = TocLogicalSectionLabel(tm, fu.frame, logical);
        fputc('{', out);
        fprintf(out,
                "\"stream_slot\":%" PRIuS ",\"logical\":%" PRIuS
                ",\"size_bytes\":%" PRIu32 ",\"byte_start\":%" PRIuS
                ",\"byte_end_exclusive\":%" PRIuS ",",
                stream, logical, sz, abs_byte, abs_byte + static_cast<size_t>(sz));
        fputs("\"label\":", out);
        JsonEscString(out, label.c_str(), label.size());
        fputc('}', out);
        body_cursor += sz;
      }
    }
    fputs("]}", out);
  }
  fputs("]}", out);
  fputs("}\n", out);
}

}  // namespace jxltran
