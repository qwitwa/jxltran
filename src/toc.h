// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Self-contained TOC (table of contents) reader/writer for jxltran:
// permutation flag, optional Lehmer-coded permutation entropy stream, byte
// padding, U32-coded section sizes, final padding. Uses bits.h
// and entropy.h only.

#ifndef TOOLS_JXLTRAN_TOC_H_
#define TOOLS_JXLTRAN_TOC_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxltran {

class BitReader;
class BitWriter;

// Reads one complete TOC from the current bit position in |br|. |sizes_opt|
// and |perm_opt| may be null (skip storing); permutation entropy is still
// parsed when the flag is set. Sets |*total| to the sum of section byte sizes.
// When |bits_before_sizes_out| is non-null, sets it to the number of bits from
// the reader position at entry through the byte padding immediately before the
// first U32-encoded section size (used to rewrite sizes while keeping the
// permutation prefix verbatim).
bool ReadFullToc(BitReader& br, size_t toc_entries, uint64_t* total,
                 std::vector<uint32_t>* sizes_opt,
                 std::vector<uint32_t>* perm_opt,
                 size_t* bits_before_sizes_out = nullptr);

// Like ReadFullToc with no size/permutation vectors (only |*total|).
bool ReadToc(BitReader& br, size_t toc_entries, uint64_t* total);

// Re-encodes a TOC at the current bit position in |bw|. If |perm| is empty,
// writes permutation_flag false, byte pad, U32-coded sizes, byte pad (same
// layout as ReadFullToc). If |perm| is non-empty, writes permutation_flag
// true, a flat-ANS permutation entropy stream (see WriteTocPermutationAnsEntropy
// in entropy.h), byte pad, then sizes.
bool WriteDecodedToc(BitWriter& bw, size_t header_bits_mod8,
                     const std::vector<uint32_t>& perm,
                     const std::vector<uint32_t>& sizes);

// Internal jxltran representation: |logical_at_stream[s]| is the logical TOC
// section index carried in stream-order slot |s| (matches libjxl dec_frame
// after grouping). The bitstream Lehmer code stores the inverse mapping
// |natural_to_stream[L]| = stream slot of logical section |L|, as in libjxl
// enc_frame WriteGroupOffsets / PermuteGroups.
bool TocPermLogicalAtStreamFromLehmerNaturalToStream(
    const std::vector<uint32_t>& natural_to_stream, size_t n,
    std::vector<uint32_t>* logical_at_stream_out);
bool TocPermLehmerNaturalToStreamFromLogicalAtStream(
    const std::vector<uint32_t>& logical_at_stream, size_t n,
    std::vector<uint32_t>* natural_to_stream_out);

// Writes only the U32-encoded section sizes and final byte padding (same as
// the tail of ReadFullToc after the permutation prefix).
bool WriteTocSizeList(BitWriter& bw, const std::vector<uint32_t>& sizes);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_TOC_H_
