// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Standalone bit-level reader and writer for JXL header parsing.
// Bits are read/written LSB-first within each byte, as specified
// in the JXL spec (Annex A "Header syntax").

#ifndef TOOLS_JXLTRAN_BITS_H_
#define TOOLS_JXLTRAN_BITS_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxltran {

// ── BitReader ────────────────────────────────────────────────────────────────

class BitReader {
 public:
  BitReader(const uint8_t* buf, size_t size_bytes)
      : buf_(buf), size_bits_(size_bytes * 8), pos_(0), ok_(true) {}

  // Read n bits (LSB first).  Returns 0 on overread (sets ok_ = false).
  uint32_t ReadBits(int n) {
    if (n == 0) return 0;
    if (pos_ + static_cast<size_t>(n) > size_bits_) {
      ok_ = false;
      return 0;
    }
    uint32_t val = 0;
    for (int i = 0; i < n; ++i, ++pos_) {
      if ((buf_[pos_ >> 3] >> (pos_ & 7)) & 1) val |= (1u << i);
    }
    return val;
  }

  // Peek up to 16 bits without advancing the position.
  uint32_t PeekBits(int n) {
    if (n == 0) return 0;
    uint32_t val = 0;
    size_t p = pos_;
    for (int i = 0; i < n && p < size_bits_; ++i, ++p)
      if ((buf_[p >> 3] >> (p & 7)) & 1) val |= (1u << i);
    return val;
  }

  // Advance position by n bits (must only be called after a matching PeekBits).
  void Consume(int n) {
    if (n > 0) pos_ += static_cast<size_t>(n);
  }

  bool ReadBool() { return ReadBits(1) != 0; }
  uint16_t ReadF16() { return static_cast<uint16_t>(ReadBits(16)); }

  // Advance to the next byte boundary (skip zero padding bits).
  void ZeroPadToByte() { pos_ = (pos_ + 7u) & ~7u; }

  // Advance |n| bits without reading them (caller must ensure the skipped
  // region was already validated, e.g. by decoding with libjxl's BitReader).
  void SkipBits(size_t n) {
    if (pos_ + n > size_bits_) {
      ok_ = false;
      pos_ = size_bits_;
      return;
    }
    pos_ += n;
  }

  // Current bit position.
  size_t pos() const { return pos_; }
  // Total bits available.
  size_t size_bits() const { return size_bits_; }
  // False if a read went past the end.
  bool ok() const { return ok_; }
  // Raw buffer pointer (for debugging).
  const uint8_t* buf() const { return buf_; }

 private:
  const uint8_t* buf_;
  size_t size_bits_;
  size_t pos_;
  bool ok_;
};

// ── BitWriter ────────────────────────────────────────────────────────────────

class BitWriter {
 public:
  // Write n bits (LSB first) from value.
  void WriteBits(int n, uint64_t value) {
    for (int i = 0; i < n; ++i, ++bit_pos_) {
      const size_t byte_idx = bit_pos_ >> 3;
      if (byte_idx >= bytes_.size()) bytes_.push_back(0);
      if ((value >> i) & 1)
        bytes_[byte_idx] |= static_cast<uint8_t>(1u << (bit_pos_ & 7));
    }
  }

  void WriteBool(bool v) { WriteBits(1, v ? 1 : 0); }
  void WriteF16(uint16_t v) { WriteBits(16, v); }

  // Pad with zero bits to the next byte boundary (padding bits must be 0).
  void ZeroPadToByte() {
    while (bit_pos_ & 7u) {
      WriteBits(1, 0);
    }
  }

  // Append all tail bits from |src| starting at bit |start_bit|.
  void AppendTailBits(const uint8_t* src, size_t src_size_bytes,
                      size_t start_bit) {
    const size_t total_bits = src_size_bytes * 8;
    for (size_t i = start_bit; i < total_bits; ++i) {
      WriteBits(1, (src[i >> 3] >> (i & 7)) & 1);
    }
  }

  // Append exactly |count| bits from |src| starting at bit |start_bit|.
  void AppendBitsRange(const uint8_t* src, size_t start_bit, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      size_t si = start_bit + i;
      WriteBits(1, (src[si >> 3] >> (si & 7)) & 1);
    }
  }

  // Append raw bytes. bit_pos_ must be byte-aligned before calling.
  void AppendRawBytes(const uint8_t* src, size_t count) {
    for (size_t i = 0; i < count; ++i) {
      bytes_.push_back(src[i]);
    }
    bit_pos_ += count * 8;
  }

  size_t bit_pos() const { return bit_pos_; }
  const std::vector<uint8_t>& bytes() const { return bytes_; }
  std::vector<uint8_t> TakeBytes() { return std::move(bytes_); }

 private:
  std::vector<uint8_t> bytes_;
  size_t bit_pos_ = 0;
};

// ── U32 ──────────────────────────────────────────────────────────────────────

// Represents one of the four distribution alternatives for U32().
struct U32Dist {
  enum Kind { kImm, kBits, kBitsOffset } kind;
  uint32_t p;  // for kImm: the value; for kBitsOffset: the additive offset
  uint32_t n;  // for kBits/kBitsOffset: number of extra bits to read/write

  static constexpr U32Dist Imm(uint32_t v) { return {kImm, v, 0}; }
  static constexpr U32Dist Bits(uint32_t n) { return {kBits, 0, n}; }
  static constexpr U32Dist BitsOffset(uint32_t n, uint32_t offset) {
    return {kBitsOffset, offset, n};
  }
};

inline uint32_t ReadU32(BitReader& br, U32Dist d0, U32Dist d1, U32Dist d2,
                        U32Dist d3) {
  const U32Dist ds[4] = {d0, d1, d2, d3};
  const auto& d = ds[br.ReadBits(2)];
  switch (d.kind) {
    case U32Dist::kImm:
      return d.p;
    case U32Dist::kBits:
      return br.ReadBits(d.n);
    case U32Dist::kBitsOffset:
      return d.p + br.ReadBits(d.n);
  }
  return 0;
}

inline void WriteU32(BitWriter& bw, uint32_t value, U32Dist d0, U32Dist d1,
                     U32Dist d2, U32Dist d3) {
  const U32Dist ds[4] = {d0, d1, d2, d3};
  for (int sel = 0; sel < 4; ++sel) {
    const auto& d = ds[sel];
    bool fits = false;
    switch (d.kind) {
      case U32Dist::kImm:
        fits = (value == d.p);
        break;
      case U32Dist::kBits:
        fits = (d.n >= 32 || value < (1u << d.n));
        break;
      case U32Dist::kBitsOffset:
        fits = (value >= d.p && (d.n >= 32 || value - d.p < (1u << d.n)));
        break;
    }
    if (fits) {
      bw.WriteBits(2, sel);
      switch (d.kind) {
        case U32Dist::kImm:
          break;
        case U32Dist::kBits:
          bw.WriteBits(d.n, value);
          break;
        case U32Dist::kBitsOffset:
          bw.WriteBits(d.n, value - d.p);
          break;
      }
      return;
    }
  }
  // Should not be reached for valid values.
}

// ── Enum ─────────────────────────────────────────────────────────────────────

// Enum(EnumTable) = U32(0, 1, 2+u(4), 18+u(6))
inline uint32_t ReadEnum(BitReader& br) {
  return ReadU32(br, U32Dist::Imm(0), U32Dist::Imm(1),
                 U32Dist::BitsOffset(4, 2), U32Dist::BitsOffset(6, 18));
}
inline void WriteEnum(BitWriter& bw, uint32_t val) {
  WriteU32(bw, val, U32Dist::Imm(0), U32Dist::Imm(1), U32Dist::BitsOffset(4, 2),
           U32Dist::BitsOffset(6, 18));
}

// ── U64 ──────────────────────────────────────────────────────────────────────

uint64_t ReadU64(BitReader& br);
void WriteU64(BitWriter& bw, uint64_t value);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_BITS_H_
