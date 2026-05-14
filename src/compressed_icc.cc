// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// ICC profile expansion (UnpredictICC) — same algorithm as lib/jxl/icc_codec.cc.

#include "compressed_icc.h"

#include <array>
#include <cstring>
#include <vector>

namespace jxltran {
namespace {

using Tag = std::array<uint8_t, 4>;

static constexpr size_t kICCHeaderSize = 128;

static const Tag kAcspTag = {{'a', 'c', 's', 'p'}};
static const Tag kBkptTag = {{'b', 'k', 'p', 't'}};
static const Tag kBtrcTag = {{'b', 'T', 'R', 'C'}};
static const Tag kBxyzTag = {{'b', 'X', 'Y', 'Z'}};
static const Tag kChadTag = {{'c', 'h', 'a', 'd'}};
static const Tag kChrmTag = {{'c', 'h', 'r', 'm'}};
static const Tag kCprtTag = {{'c', 'p', 'r', 't'}};
static const Tag kCurvTag = {{'c', 'u', 'r', 'v'}};
static const Tag kDescTag = {{'d', 'e', 's', 'c'}};
static const Tag kDmddTag = {{'d', 'm', 'd', 'd'}};
static const Tag kDmndTag = {{'d', 'm', 'n', 'd'}};
static const Tag kGbd_Tag = {{'g', 'b', 'd', ' '}};
static const Tag kGtrcTag = {{'g', 'T', 'R', 'C'}};
static const Tag kGxyzTag = {{'g', 'X', 'Y', 'Z'}};
static const Tag kKtrcTag = {{'k', 'T', 'R', 'C'}};
static const Tag kKxyzTag = {{'k', 'X', 'Y', 'Z'}};
static const Tag kLumiTag = {{'l', 'u', 'm', 'i'}};
static const Tag kMab_Tag = {{'m', 'A', 'B', ' '}};
static const Tag kMba_Tag = {{'m', 'B', 'A', ' '}};
static const Tag kMlucTag = {{'m', 'l', 'u', 'c'}};
static const Tag kMntrTag = {{'m', 'n', 't', 'r'}};
static const Tag kParaTag = {{'p', 'a', 'r', 'a'}};
static const Tag kRgb_Tag = {{'R', 'G', 'B', ' '}};
static const Tag kRtrcTag = {{'r', 'T', 'R', 'C'}};
static const Tag kRxyzTag = {{'r', 'X', 'Y', 'Z'}};
static const Tag kSf32Tag = {{'s', 'f', '3', '2'}};
static const Tag kTextTag = {{'t', 'e', 'x', 't'}};
static const Tag kVcgtTag = {{'v', 'c', 'g', 't'}};
static const Tag kWtptTag = {{'w', 't', 'p', 't'}};
static const Tag kXyz_Tag = {{'X', 'Y', 'Z', ' '}};

static constexpr size_t kNumTagStrings = 17;
static const Tag* const kTagStrings[kNumTagStrings] = {
    &kCprtTag, &kWtptTag, &kBkptTag, &kRxyzTag, &kGxyzTag, &kBxyzTag,
    &kKxyzTag, &kRtrcTag, &kGtrcTag, &kBtrcTag, &kKtrcTag, &kChadTag,
    &kDescTag, &kChrmTag, &kDmndTag, &kDmddTag, &kLumiTag};

static constexpr size_t kCommandTagUnknown = 1;
static constexpr size_t kCommandTagTRC = 2;
static constexpr size_t kCommandTagXYZ = 3;
static constexpr size_t kCommandTagStringFirst = 4;

static constexpr size_t kNumTypeStrings = 8;
static const Tag* const kTypeStrings[kNumTypeStrings] = {
    &kXyz_Tag, &kDescTag, &kTextTag, &kMlucTag,
    &kParaTag, &kCurvTag, &kSf32Tag, &kGbd_Tag};

static constexpr size_t kCommandInsert = 1;
static constexpr size_t kCommandShuffle2 = 2;
static constexpr size_t kCommandShuffle4 = 3;
static constexpr size_t kCommandPredict = 4;
static constexpr size_t kCommandXYZ = 10;
static constexpr size_t kCommandTypeStartFirst = 16;

static constexpr size_t kFlagBitOffset = 64;
static constexpr size_t kFlagBitSize = 128;

static void StoreBE32(uint32_t v, uint8_t* p) {
  p[0] = static_cast<uint8_t>((v >> 24) & 0xFF);
  p[1] = static_cast<uint8_t>((v >> 16) & 0xFF);
  p[2] = static_cast<uint8_t>((v >> 8) & 0xFF);
  p[3] = static_cast<uint8_t>(v & 0xFF);
}

static uint32_t LoadBE32(const uint8_t* p) {
  return (uint32_t(p[0]) << 24) | (uint32_t(p[1]) << 16) |
         (uint32_t(p[2]) << 8) | uint32_t(p[3]);
}

static uint32_t DecodeUint32(const uint8_t* data, size_t size, size_t pos) {
  return pos + 4 > size ? 0 : LoadBE32(data + pos);
}

static bool CheckOutOfBounds(uint64_t a, uint64_t b, uint64_t size) {
  uint64_t pos = a + b;
  if (pos > size) return false;
  if (pos < a) return false;
  return true;
}

static bool CheckIs32Bit(uint64_t v) {
  constexpr uint64_t kUpper32 = ~static_cast<uint64_t>(0xFFFFFFFF);
  return (v & kUpper32) == 0;
}

static void AppendUint32(uint32_t value, std::vector<uint8_t>* data) {
  size_t pos = data->size();
  data->resize(pos + 4);
  StoreBE32(value, data->data() + pos);
}

static Tag DecodeKeyword(const uint8_t* data, size_t size, size_t pos) {
  if (pos + 4 > size) return {{' ', ' ', ' ', ' '}};
  return {{data[pos], data[pos + 1], data[pos + 2], data[pos + 3]}};
}

static void AppendKeyword(const Tag& keyword, std::vector<uint8_t>* data) {
  data->insert(data->end(), keyword.begin(), keyword.end());
}

static uint64_t DecodeVarInt(const uint8_t* input, size_t inputSize,
                             size_t* pos) {
  size_t i;
  uint64_t ret = 0;
  for (i = 0; *pos + i < inputSize && i < 10; ++i) {
    ret |= static_cast<uint64_t>(input[*pos + i] & 127)
           << static_cast<uint64_t>(7 * i);
    if ((input[*pos + i] & 128) == 0) break;
  }
  *pos += i + 1;
  return ret;
}

static bool Shuffle(uint8_t* data, size_t size, size_t width) {
  size_t height = (size + width - 1) / width;
  std::vector<uint8_t> result(size);
  size_t s = 0;
  size_t j = 0;
  for (size_t i = 0; i < size; i++) {
    result[i] = data[j];
    j += height;
    if (j >= size) j = ++s;
  }
  std::memcpy(data, result.data(), size);
  return true;
}

static const std::array<uint8_t, kICCHeaderSize> kIccInitialHeaderPrediction = {
    {0,   0,   0,   0,   0,   0,   0,   0,   4, 0, 0, 0, 'm', 'n', 't', 'r',
     'R', 'G', 'B', ' ', 'X', 'Y', 'Z', ' ', 0, 0, 0, 0, 0,   0,   0,   0,
     0,   0,   0,   0,   'a', 'c', 's', 'p', 0, 0, 0, 0, 0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   246, 214, 0, 1, 0, 0, 0,   0,   211, 45,
     0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0,
     0,   0,   0,   0,   0,   0,   0,   0,   0, 0, 0, 0, 0,   0,   0,   0}};

static std::array<uint8_t, kICCHeaderSize> ICCInitialHeaderPrediction(
    uint32_t size) {
  std::array<uint8_t, kICCHeaderSize> copy = kIccInitialHeaderPrediction;
  StoreBE32(size, copy.data());
  return copy;
}

static void ICCPredictHeader(const uint8_t* icc, size_t size, uint8_t* header,
                             size_t pos) {
  if (pos == 8 && size >= 8) {
    header[80] = icc[4];
    header[81] = icc[5];
    header[82] = icc[6];
    header[83] = icc[7];
  }
  if (pos == 41 && size >= 41) {
    if (icc[40] == 'A') {
      header[41] = 'P';
      header[42] = 'P';
      header[43] = 'L';
    }
    if (icc[40] == 'M') {
      header[41] = 'S';
      header[42] = 'F';
      header[43] = 'T';
    }
  }
  if (pos == 42 && size >= 42) {
    if (icc[40] == 'S' && icc[41] == 'G') {
      header[42] = 'I';
      header[43] = ' ';
    }
    if (icc[40] == 'S' && icc[41] == 'U') {
      header[42] = 'N';
      header[43] = 'W';
    }
  }
}

template <typename T>
static T PredictValue(T p1, T p2, T p3, int order) {
  if (order == 0) return p1;
  if (order == 1) return 2 * p1 - p2;
  if (order == 2) return 3 * p1 - 3 * p2 + p3;
  return 0;
}

static uint8_t LinearPredictICCValue(const uint8_t* data, size_t start,
                                     size_t i, size_t stride, size_t width,
                                     int order) {
  size_t pos = start + i;
  if (width == 1) {
    uint8_t p1 = data[pos - stride];
    uint8_t p2 = data[pos - stride * 2];
    uint8_t p3 = data[pos - stride * 3];
    return PredictValue(p1, p2, p3, order);
  }
  if (width == 2) {
    size_t p = start + (i & ~1ull);
    uint16_t p1 =
        static_cast<uint16_t>((data[p - stride * 1] << 8) + data[p - stride * 1 + 1]);
    uint16_t p2 =
        static_cast<uint16_t>((data[p - stride * 2] << 8) + data[p - stride * 2 + 1]);
    uint16_t p3 =
        static_cast<uint16_t>((data[p - stride * 3] << 8) + data[p - stride * 3 + 1]);
    uint16_t pred = PredictValue(p1, p2, p3, order);
    return (i & 1) ? static_cast<uint8_t>(pred & 255)
                   : static_cast<uint8_t>((pred >> 8) & 255);
  }
  size_t p = start + (i & ~3ull);
  uint32_t p1 = DecodeUint32(data, pos, p - stride);
  uint32_t p2 = DecodeUint32(data, pos, p - stride * 2);
  uint32_t p3 = DecodeUint32(data, pos, p - stride * 3);
  uint32_t pred = PredictValue(p1, p2, p3, order);
  unsigned shiftbytes = 3 - static_cast<unsigned>(i & 3);
  return static_cast<uint8_t>((pred >> (shiftbytes * 8)) & 255);
}

}  // namespace

bool UnpredictIccProfile(const uint8_t* enc, size_t size,
                         std::vector<uint8_t>* result) {
  if (!result->empty()) return false;
  size_t pos = 0;
  if (pos >= size) return false;
  uint64_t osize = DecodeVarInt(enc, size, &pos);
  if (!CheckIs32Bit(osize)) return false;
  if (pos >= size) return false;
  uint64_t csize = DecodeVarInt(enc, size, &pos);
  if (!CheckIs32Bit(csize)) return false;
  size_t cpos = pos;
  if (!CheckOutOfBounds(pos, csize, size)) return false;
  size_t commands_end = cpos + csize;
  pos = commands_end;

  std::array<uint8_t, kICCHeaderSize> header = ICCInitialHeaderPrediction(
      static_cast<uint32_t>(osize));
  for (size_t i = 0; i <= kICCHeaderSize; i++) {
    if (result->size() == osize) {
      if (cpos != commands_end) return false;
      if (pos != size) return false;
      return true;
    }
    if (i == kICCHeaderSize) break;
    ICCPredictHeader(result->data(), result->size(), header.data(), i);
    if (pos >= size) return false;
    result->push_back(static_cast<uint8_t>(enc[pos++] + header[i]));
  }
  if (cpos >= commands_end) return false;

  uint64_t numtags = DecodeVarInt(enc, size, &cpos);

  if (numtags != 0) {
    numtags--;
    if (!CheckIs32Bit(numtags)) return false;
    AppendUint32(static_cast<uint32_t>(numtags), result);
    uint64_t prevtagstart = kICCHeaderSize + numtags * 12;
    uint64_t prevtagsize = 0;
    for (;;) {
      if (result->size() > osize) return false;
      if (cpos > commands_end) return false;
      if (cpos == commands_end) break;
      uint8_t command = enc[cpos++];
      uint8_t tagcode = command & 63;
      Tag tag;
      if (tagcode == 0) {
        break;
      }
      if (tagcode == kCommandTagUnknown) {
        if (!CheckOutOfBounds(pos, 4, size)) return false;
        tag = DecodeKeyword(enc, size, pos);
        pos += 4;
      } else if (tagcode == kCommandTagTRC) {
        tag = kRtrcTag;
      } else if (tagcode == kCommandTagXYZ) {
        tag = kRxyzTag;
      } else {
        if (tagcode - kCommandTagStringFirst >= kNumTagStrings) return false;
        tag = *kTagStrings[tagcode - kCommandTagStringFirst];
      }
      AppendKeyword(tag, result);

      uint64_t tagstart;
      uint64_t tagsize = prevtagsize;
      if (tag == kRxyzTag || tag == kGxyzTag || tag == kBxyzTag ||
          tag == kKxyzTag || tag == kWtptTag || tag == kBkptTag ||
          tag == kLumiTag) {
        tagsize = 20;
      }

      if (command & kFlagBitOffset) {
        if (cpos >= commands_end) return false;
        tagstart = DecodeVarInt(enc, size, &cpos);
      } else {
        if (!CheckIs32Bit(prevtagstart)) return false;
        tagstart = prevtagstart + prevtagsize;
      }
      if (!CheckIs32Bit(tagstart)) return false;
      AppendUint32(static_cast<uint32_t>(tagstart), result);
      if (command & kFlagBitSize) {
        if (cpos >= commands_end) return false;
        tagsize = DecodeVarInt(enc, size, &cpos);
      }
      if (!CheckIs32Bit(tagsize)) return false;
      AppendUint32(static_cast<uint32_t>(tagsize), result);
      prevtagstart = tagstart;
      prevtagsize = tagsize;

      if (tagcode == kCommandTagTRC) {
        AppendKeyword(kGtrcTag, result);
        AppendUint32(static_cast<uint32_t>(tagstart), result);
        AppendUint32(static_cast<uint32_t>(tagsize), result);
        AppendKeyword(kBtrcTag, result);
        AppendUint32(static_cast<uint32_t>(tagstart), result);
        AppendUint32(static_cast<uint32_t>(tagsize), result);
      }

      if (tagcode == kCommandTagXYZ) {
        if (!CheckIs32Bit(tagstart + tagsize * 2)) return false;
        AppendKeyword(kGxyzTag, result);
        AppendUint32(static_cast<uint32_t>(tagstart + tagsize), result);
        AppendUint32(static_cast<uint32_t>(tagsize), result);
        AppendKeyword(kBxyzTag, result);
        AppendUint32(static_cast<uint32_t>(tagstart + tagsize * 2), result);
        AppendUint32(static_cast<uint32_t>(tagsize), result);
      }
    }
  }

  for (;;) {
    if (result->size() > osize) return false;
    if (cpos > commands_end) return false;
    if (cpos == commands_end) break;
    uint8_t command = enc[cpos++];
    if (command == kCommandInsert) {
      if (cpos >= commands_end) return false;
      uint64_t num = DecodeVarInt(enc, size, &cpos);
      if (!CheckOutOfBounds(pos, num, size)) return false;
      for (size_t i = 0; i < num; i++) {
        result->push_back(enc[pos++]);
      }
    } else if (command == kCommandShuffle2 || command == kCommandShuffle4) {
      if (cpos >= commands_end) return false;
      uint64_t num = DecodeVarInt(enc, size, &cpos);
      if (!CheckOutOfBounds(pos, num, size)) return false;
      std::vector<uint8_t> shuffled(num);
      for (size_t i = 0; i < num; i++) {
        shuffled[i] = enc[pos + i];
      }
      if (command == kCommandShuffle2) {
        if (!Shuffle(shuffled.data(), num, 2)) return false;
      } else {
        if (!Shuffle(shuffled.data(), num, 4)) return false;
      }
      for (size_t i = 0; i < num; i++) {
        result->push_back(shuffled[i]);
        pos++;
      }
    } else if (command == kCommandPredict) {
      if (!CheckOutOfBounds(cpos, 2, commands_end)) return false;
      uint8_t flags = enc[cpos++];

      size_t width = (flags & 3) + 1;
      if (width == 3) return false;

      int order = (flags & 12) >> 2;
      if (order == 3) return false;

      uint64_t stride = width;
      if (flags & 16) {
        if (cpos >= commands_end) return false;
        stride = DecodeVarInt(enc, size, &cpos);
        if (stride < width) return false;
      }
      if (result->empty() || ((result->size() - 1u) >> 2u) < stride) {
        return false;
      }

      if (cpos >= commands_end) return false;
      uint64_t num = DecodeVarInt(enc, size, &cpos);
      if (!CheckOutOfBounds(pos, num, size)) return false;

      std::vector<uint8_t> shuffled(num);
      for (size_t i = 0; i < num; i++) {
        shuffled[i] = enc[pos + i];
      }
      if (width > 1) {
        if (!Shuffle(shuffled.data(), num, width)) return false;
      }

      size_t start = result->size();
      for (size_t i = 0; i < num; i++) {
        uint8_t predicted = LinearPredictICCValue(result->data(), start, i,
                                                  stride, width, order);
        result->push_back(static_cast<uint8_t>(predicted + shuffled[i]));
      }
      pos += num;
    } else if (command == kCommandXYZ) {
      AppendKeyword(kXyz_Tag, result);
      for (int i = 0; i < 4; i++) {
        result->push_back(0);
      }
      if (!CheckOutOfBounds(pos, 12, size)) return false;
      for (size_t i = 0; i < 12; i++) {
        result->push_back(enc[pos++]);
      }
    } else if (command >= kCommandTypeStartFirst &&
               command < kCommandTypeStartFirst + kNumTypeStrings) {
      AppendKeyword(*kTypeStrings[command - kCommandTypeStartFirst], result);
      for (size_t i = 0; i < 4; i++) {
        result->push_back(0);
      }
    } else {
      return false;
    }
  }

  if (pos != size) return false;
  if (result->size() != osize) return false;
  return true;
}

}  // namespace jxltran
