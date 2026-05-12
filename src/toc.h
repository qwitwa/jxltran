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
bool ReadFullToc(BitReader& br, size_t toc_entries, uint64_t* total,
                 std::vector<uint32_t>* sizes_opt,
                 std::vector<uint32_t>* perm_opt);

// Like ReadFullToc with no size/permutation vectors (only |*total|).
bool ReadToc(BitReader& br, size_t toc_entries, uint64_t* total);

// Re-encodes a TOC at the current bit position in |bw| (permutation flag
// false, byte pad, U32-coded sizes, byte pad — same layout as ReadFullToc).
// The second parameter is unused (kept for a stable call signature).
bool WriteDecodedToc(BitWriter& bw, size_t header_bits_mod8,
                     const std::vector<uint32_t>& perm,
                     const std::vector<uint32_t>& sizes);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_TOC_H_
