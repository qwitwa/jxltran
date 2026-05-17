// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "box.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <map>

#ifdef JXLTRAN_HAVE_BROTLI
#include <brotli/decode.h>
#include <brotli/encode.h>
#endif

namespace jxltran {

namespace {

uint32_t LoadBE32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

uint64_t LoadBE64(const uint8_t* p) {
  return (uint64_t(LoadBE32(p)) << 32) | uint64_t(LoadBE32(p + 4));
}

void StoreBE32(uint32_t v, uint8_t* p) {
  p[0] = (v >> 24) & 0xFF;
  p[1] = (v >> 16) & 0xFF;
  p[2] = (v >> 8) & 0xFF;
  p[3] = v & 0xFF;
}

}  // namespace

bool ParseBoxes(const uint8_t* buf, size_t size, std::vector<JxlBox>* out) {
  size_t pos = 0;
  while (pos < size) {
    if (pos + 8 > size) {
      fprintf(stderr, "Truncated box header at offset %zu\n", pos);
      return false;
    }
    uint64_t box_size = LoadBE32(buf + pos);
    const uint8_t* type_ptr = buf + pos + 4;
    size_t header_size = 8;
    if (box_size == 1) {
      if (pos + 16 > size) {
        fprintf(stderr, "Truncated extended box header at offset %zu\n", pos);
        return false;
      }
      box_size = LoadBE64(buf + pos + 8);
      header_size = 16;
    }
    size_t data_size;
    if (box_size == 0) {
      // Box extends to end of file.
      data_size = size - pos - header_size;
      box_size = size - pos;
    } else {
      if (box_size < header_size) {
        fprintf(stderr, "Box size %llu too small at offset %zu\n",
                (unsigned long long)box_size, pos);
        return false;
      }
      data_size = box_size - header_size;
    }
    if (pos + box_size > size) {
      fprintf(stderr, "Box extends past end of file at offset %zu\n", pos);
      return false;
    }
    JxlBox box;
    memcpy(box.type, type_ptr, 4);
    box.file_offset = pos;
    box.file_total_bytes = static_cast<size_t>(box_size);
    box.header_size = static_cast<uint8_t>(header_size);
    box.data.assign(buf + pos + header_size,
                    buf + pos + header_size + data_size);
    out->push_back(std::move(box));
    pos += box_size;
  }
  return true;
}

void SerializeBox(const JxlBox& box, std::vector<uint8_t>* out) {
  uint64_t total = 8 + box.data.size();
  uint8_t header[8];
  if (total > 0xFFFFFFFFU) {
    // Extended 64-bit size.
    out->resize(out->size() + 16 + box.data.size());
    uint8_t* p = out->data() + out->size() - 16 - box.data.size();
    StoreBE32(1, p);
    memcpy(p + 4, box.type, 4);
    // Store 64-bit size at p+8.
    uint64_t ext = 16 + box.data.size();
    for (int i = 7; i >= 0; --i) {
      p[8 + i] = ext & 0xFF;
      ext >>= 8;
    }
    memcpy(p + 16, box.data.data(), box.data.size());
  } else {
    StoreBE32(static_cast<uint32_t>(total), header);
    memcpy(header + 4, box.type, 4);
    out->insert(out->end(), header, header + 8);
    out->insert(out->end(), box.data.begin(), box.data.end());
  }
}

void WriteJxlSignatureBox(std::vector<uint8_t>* out) {
  // Size=12, type="JXL ", data="\r\n\x87\n"
  static const uint8_t kBox[] = {0,   0,   0,   12,  'J',  'X',
                                 'L', ' ', 0xd, 0xa, 0x87, 0xa};
  out->insert(out->end(), kBox, kBox + sizeof(kBox));
}

void WriteJxlFtypBox(std::vector<uint8_t>* out) {
  // Size=20, type="ftyp", major_brand="jxl ", minor_version=0,
  // compatible_brands=["jxl "]
  static const uint8_t kBox[] = {0,   0,   0, 20, 'f', 't', 'y', 'p', 'j', 'x',
                                 'l', ' ', 0, 0,  0,   0,   'j', 'x', 'l', ' '};
  out->insert(out->end(), kBox, kBox + sizeof(kBox));
}

bool IsJxlContainer(const uint8_t* buf, size_t size) {
  static const uint8_t kMagic[] = {0,   0,   0,   12,  'J',  'X',
                                   'L', ' ', 0xd, 0xa, 0x87, 0xa};
  return size >= sizeof(kMagic) && memcmp(buf, kMagic, sizeof(kMagic)) == 0;
}

bool IsJxlCodestream(const uint8_t* buf, size_t size) {
  return size >= 2 && buf[0] == 0xFF && buf[1] == 0x0A;
}

bool HasJxlpBoxes(const std::vector<JxlBox>& boxes) {
  for (const auto& box : boxes) {
    if (memcmp(box.type, "jxlp", 4) == 0) return true;
  }
  return false;
}

bool ReassembleCodestream(const std::vector<JxlBox>& boxes,
                          std::vector<uint8_t>* codestream) {
  bool has_jxlc = false;
  bool has_jxlp = false;
  for (const auto& box : boxes) {
    if (memcmp(box.type, "jxlc", 4) == 0) has_jxlc = true;
    if (memcmp(box.type, "jxlp", 4) == 0) has_jxlp = true;
  }
  if (has_jxlc && has_jxlp) {
    fprintf(stderr, "Error: mixed jxlc and jxlp boxes\n");
    return false;
  }
  if (has_jxlc) {
    for (const auto& box : boxes) {
      if (memcmp(box.type, "jxlc", 4) == 0) {
        codestream->insert(codestream->end(), box.data.begin(), box.data.end());
      }
    }
    return true;
  }
  if (has_jxlp) {
    // Collect jxlp boxes, sort by counter (low 31 bits of first 4 bytes).
    std::map<uint32_t, const std::vector<uint8_t>*> by_index;
    bool seen_last = false;
    for (const auto& box : boxes) {
      if (memcmp(box.type, "jxlp", 4) != 0) continue;
      if (box.data.size() < 4) {
        fprintf(stderr, "jxlp box too small\n");
        return false;
      }
      uint32_t word = LoadBE32(box.data.data());
      uint32_t index = word & 0x7FFFFFFF;
      bool is_last = (word & 0x80000000) != 0;
      if (by_index.count(index)) {
        fprintf(stderr, "Duplicate jxlp counter %u\n", index);
        return false;
      }
      by_index[index] = &box.data;
      if (is_last) seen_last = true;
    }
    if (!seen_last) {
      fprintf(stderr, "Warning: no jxlp box with last-box flag set\n");
    }
    for (const auto& kv : by_index) {
      const auto& data = *kv.second;
      // Skip the 4-byte counter header.
      codestream->insert(codestream->end(), data.begin() + 4, data.end());
    }
    return true;
  }
  fprintf(stderr, "Error: no jxlc or jxlp boxes found\n");
  return false;
}

bool HasMetadataBoxes(const std::vector<JxlBox>& boxes) {
  for (const auto& box : boxes) {
    if (memcmp(box.type, "JXL ", 4) == 0) continue;
    if (memcmp(box.type, "ftyp", 4) == 0) continue;
    if (memcmp(box.type, "jxll", 4) == 0) continue;
    if (memcmp(box.type, "jxlc", 4) == 0) continue;
    if (memcmp(box.type, "jxlp", 4) == 0) continue;
    return true;
  }
  return false;
}

bool SortJxlpBoxes(std::vector<JxlBox>* boxes) {
  // Collect jxlp payloads indexed by counter.
  std::map<uint32_t, std::vector<uint8_t>> by_index;
  for (const auto& box : *boxes) {
    if (memcmp(box.type, "jxlp", 4) != 0) continue;
    if (box.data.size() < 4) {
      fprintf(stderr, "jxlp box too small\n");
      return false;
    }
    uint32_t index = LoadBE32(box.data.data()) & 0x7FFFFFFF;
    if (by_index.count(index)) {
      fprintf(stderr, "Duplicate jxlp counter %u\n", index);
      return false;
    }
    by_index[index] = {box.data.begin() + 4, box.data.end()};
  }
  if (by_index.empty()) return true;

  // Split into prefix (before first codestream box) and suffix (after last).
  std::vector<JxlBox> prefix, suffix;
  bool in_cs = false;
  for (const auto& box : *boxes) {
    bool is_cs =
        memcmp(box.type, "jxlp", 4) == 0 || memcmp(box.type, "jxlc", 4) == 0;
    if (is_cs) {
      in_cs = true;
    } else if (!in_cs) {
      prefix.push_back(box);
    } else {
      suffix.push_back(box);
    }
  }

  uint32_t last_index = by_index.rbegin()->first;
  boxes->clear();
  for (auto& b : prefix) boxes->push_back(std::move(b));
  for (auto& kv : by_index) {
    JxlBox jxlp;
    memcpy(jxlp.type, "jxlp", 4);
    bool is_last = (kv.first == last_index);
    uint32_t word = kv.first | (is_last ? 0x80000000u : 0u);
    jxlp.data.resize(4 + kv.second.size());
    StoreBE32(word, jxlp.data.data());
    memcpy(jxlp.data.data() + 4, kv.second.data(), kv.second.size());
    boxes->push_back(std::move(jxlp));
  }
  for (auto& b : suffix) boxes->push_back(std::move(b));
  return true;
}

bool MergeJxlpBoxes(std::vector<JxlBox>* boxes) {
  std::vector<uint8_t> codestream;
  if (!ReassembleCodestream(*boxes, &codestream)) return false;
  std::vector<JxlBox> result;
  bool inserted = false;
  for (const auto& box : *boxes) {
    bool is_cs =
        memcmp(box.type, "jxlc", 4) == 0 || memcmp(box.type, "jxlp", 4) == 0;
    if (is_cs) {
      if (!inserted) {
        JxlBox jxlc;
        memcpy(jxlc.type, "jxlc", 4);
        jxlc.data = std::move(codestream);
        result.push_back(std::move(jxlc));
        inserted = true;
      }
    } else {
      result.push_back(box);
    }
  }
  *boxes = std::move(result);
  return true;
}

static bool IsStructural(const char* type) {
  return memcmp(type, "JXL ", 4) == 0 || memcmp(type, "ftyp", 4) == 0 ||
         memcmp(type, "jxll", 4) == 0 || memcmp(type, "jxlc", 4) == 0 ||
         memcmp(type, "jxlp", 4) == 0;
}

static bool IsCodestream(const char* type) {
  return memcmp(type, "jxlc", 4) == 0 || memcmp(type, "jxlp", 4) == 0;
}

static bool MatchesStripFlag(const JxlBox& box, uint32_t flags) {
  if (IsStructural(box.type)) return false;
  if (flags == kStripAll) return true;
  // For brob boxes check the wrapped type (first 4 payload bytes).
  const char* effective_type = box.type;
  char brob_inner[4];
  if (memcmp(box.type, "brob", 4) == 0 && box.data.size() >= 4) {
    memcpy(brob_inner, box.data.data(), 4);
    effective_type = brob_inner;
  }
  if ((flags & kStripExif) && memcmp(effective_type, "Exif", 4) == 0)
    return true;
  if ((flags & kStripXmp) && memcmp(effective_type, "xml ", 4) == 0)
    return true;
  if ((flags & kStripJumbf) && memcmp(effective_type, "jumb", 4) == 0)
    return true;
  if ((flags & kStripJbrd) && memcmp(effective_type, "jbrd", 4) == 0)
    return true;
  return false;
}

void StripBoxesByType(std::vector<JxlBox>* boxes, uint32_t flags) {
  if (flags == 0) return;
  boxes->erase(std::remove_if(
                   boxes->begin(), boxes->end(),
                   [&](const JxlBox& b) { return MatchesStripFlag(b, flags); }),
               boxes->end());
}

void ReorderBoxes(std::vector<JxlBox>* boxes, BoxOrder order,
                  const std::vector<std::array<char, 4>>& explicit_order) {
  if (order == BoxOrder::kAsIs) return;
  std::vector<JxlBox> prefix, pool;
  for (auto& box : *boxes) {
    if (IsStructural(box.type)) {
      prefix.push_back(std::move(box));
    } else {
      pool.push_back(std::move(box));
    }
  }
  boxes->clear();
  for (auto& b : prefix) boxes->push_back(std::move(b));

  if (order == BoxOrder::kExplicit) {
    std::vector<bool> used(pool.size(), false);
    for (const auto& t : explicit_order) {
      for (size_t i = 0; i < pool.size(); ++i) {
        if (!used[i] && memcmp(pool[i].type, t.data(), 4) == 0) {
          boxes->push_back(std::move(pool[i]));
          used[i] = true;
        }
      }
    }
    for (size_t i = 0; i < pool.size(); ++i) {
      if (!used[i]) boxes->push_back(std::move(pool[i]));
    }
    return;
  }

  // kBeforeCodestream or kAfterCodestream: split pool into cs and meta.
  std::vector<JxlBox> cs, meta;
  for (auto& b : pool) {
    if (IsCodestream(b.type)) {
      cs.push_back(std::move(b));
    } else {
      meta.push_back(std::move(b));
    }
  }
  if (order == BoxOrder::kAfterCodestream) {
    for (auto& b : cs) boxes->push_back(std::move(b));
    for (auto& b : meta) boxes->push_back(std::move(b));
  } else {  // kBeforeCodestream
    for (auto& b : meta) boxes->push_back(std::move(b));
    for (auto& b : cs) boxes->push_back(std::move(b));
  }
}

#ifdef JXLTRAN_HAVE_BROTLI

static bool BrotliDecompress(const uint8_t* in, size_t in_size,
                             std::vector<uint8_t>* out) {
  BrotliDecoderState* st =
      BrotliDecoderCreateInstance(nullptr, nullptr, nullptr);
  if (!st) return false;
  out->resize(std::max(in_size * 4, size_t{4096}));
  size_t avail_in = in_size;
  const uint8_t* next_in = in;
  size_t total = 0;
  BrotliDecoderResult r;
  do {
    size_t avail_out = out->size() - total;
    uint8_t* next_out = out->data() + total;
    r = BrotliDecoderDecompressStream(st, &avail_in, &next_in, &avail_out,
                                        &next_out, &total);
    if (r == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT)
      out->resize(out->size() * 2);
  } while (r == BROTLI_DECODER_RESULT_NEEDS_MORE_OUTPUT);
  BrotliDecoderDestroyInstance(st);
  if (r != BROTLI_DECODER_RESULT_SUCCESS) {
    fprintf(stderr, "Brotli decompression failed\n");
    return false;
  }
  out->resize(total);
  return true;
}

static bool BrotliCompress(const uint8_t* in, size_t in_size,
                           std::vector<uint8_t>* out) {
  size_t max_out = BrotliEncoderMaxCompressedSize(in_size);
  out->resize(max_out);
  size_t out_size = max_out;
  if (!BrotliEncoderCompress(BROTLI_MAX_QUALITY, BROTLI_DEFAULT_WINDOW,
                             BROTLI_MODE_GENERIC, in_size, in, &out_size,
                             out->data())) {
    fprintf(stderr, "Brotli compression failed\n");
    return false;
  }
  out->resize(out_size);
  return true;
}

bool DecompressBrobBoxes(std::vector<JxlBox>* boxes) {
  for (auto& box : *boxes) {
    if (memcmp(box.type, "brob", 4) != 0) continue;
    if (box.data.size() < 4) {
      fprintf(stderr, "brob box too small\n");
      return false;
    }
    char inner_type[4];
    memcpy(inner_type, box.data.data(), 4);
    std::vector<uint8_t> decompressed;
    if (!BrotliDecompress(box.data.data() + 4, box.data.size() - 4,
                          &decompressed)) {
      return false;
    }
    memcpy(box.type, inner_type, 4);
    box.data = std::move(decompressed);
  }
  return true;
}

bool CompressMetadataBoxes(std::vector<JxlBox>* boxes,
                           const std::vector<std::array<char, 4>>& types) {
  static const char* kDefaultCompressible[] = {"Exif", "xml ", "jumb", nullptr};
  for (auto& box : *boxes) {
    if (memcmp(box.type, "brob", 4) == 0) continue;
    bool eligible = false;
    if (types.empty()) {
      for (int i = 0; kDefaultCompressible[i]; ++i) {
        if (memcmp(box.type, kDefaultCompressible[i], 4) == 0) {
          eligible = true;
          break;
        }
      }
    } else {
      for (const auto& t : types) {
        if (memcmp(box.type, t.data(), 4) == 0) {
          eligible = true;
          break;
        }
      }
    }
    if (!eligible) continue;
    std::vector<uint8_t> compressed;
    if (!BrotliCompress(box.data.data(), box.data.size(), &compressed)) {
      return false;
    }
    JxlBox brob;
    memcpy(brob.type, "brob", 4);
    brob.data.resize(4 + compressed.size());
    memcpy(brob.data.data(), box.type, 4);
    memcpy(brob.data.data() + 4, compressed.data(), compressed.size());
    box = std::move(brob);
  }
  return true;
}

bool ExtractMetadataPayloadToBuffer(const std::vector<JxlBox>& boxes,
                                    const char want_type[4],
                                    std::vector<uint8_t>* payload) {
  payload->clear();
  for (const JxlBox& box : boxes) {
    const char* eff = box.type;
    const uint8_t* brob_in = nullptr;
    size_t brob_len = 0;
    if (memcmp(box.type, "brob", 4) == 0) {
      if (box.data.size() < 4) continue;
      eff = reinterpret_cast<const char*>(box.data.data());
      brob_in = box.data.data() + 4;
      brob_len = box.data.size() - 4;
    }
    if (memcmp(eff, want_type, 4) != 0) continue;
    if (memcmp(box.type, "brob", 4) == 0) {
      if (!BrotliDecompress(brob_in, brob_len, payload)) return false;
    } else {
      *payload = box.data;
    }
    return true;
  }
  fprintf(stderr,
          "jxltran: no metadata box of type '%.4s' found in container\n",
          want_type);
  return false;
}

#endif  // JXLTRAN_HAVE_BROTLI

#ifndef JXLTRAN_HAVE_BROTLI

bool ExtractMetadataPayloadToBuffer(const std::vector<JxlBox>& boxes,
                                    const char want_type[4],
                                    std::vector<uint8_t>* payload) {
  payload->clear();
  for (const JxlBox& box : boxes) {
    const char* eff = box.type;
    if (memcmp(box.type, "brob", 4) == 0) {
      if (box.data.size() < 4) continue;
      eff = reinterpret_cast<const char*>(box.data.data());
    }
    if (memcmp(eff, want_type, 4) != 0) continue;
    if (memcmp(box.type, "brob", 4) == 0) {
      fprintf(stderr,
              "jxltran: metadata is inside a brob box; rebuild jxltran with "
              "brotli (libbrotli) to decompress it.\n");
      return false;
    }
    *payload = box.data;
    return true;
  }
  fprintf(stderr,
          "jxltran: no metadata box of type '%.4s' found in container\n",
          want_type);
  return false;
}

#endif  // !JXLTRAN_HAVE_BROTLI

void ReplaceMetadataBox(std::vector<JxlBox>* boxes, const char target_type[4],
                        std::vector<uint8_t> payload) {
  uint32_t strip_flag = 0;
  if (memcmp(target_type, "Exif", 4) == 0) {
    strip_flag = kStripExif;
  } else if (memcmp(target_type, "xml ", 4) == 0) {
    strip_flag = kStripXmp;
  } else if (memcmp(target_type, "jumb", 4) == 0) {
    strip_flag = kStripJumbf;
  } else {
    return;
  }
  StripBoxesByType(boxes, strip_flag);
  JxlBox nb;
  memcpy(nb.type, target_type, 4);
  nb.data = std::move(payload);
  for (size_t i = 0; i < boxes->size(); ++i) {
    if (memcmp((*boxes)[i].type, "jxlc", 4) == 0 ||
        memcmp((*boxes)[i].type, "jxlp", 4) == 0) {
      boxes->insert(boxes->begin() + i, std::move(nb));
      return;
    }
  }
  boxes->push_back(std::move(nb));
}

}  // namespace jxltran
