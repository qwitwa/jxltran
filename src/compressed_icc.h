// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Decompresses the ICC profile bytes produced by the JPEG XL ICC entropy
// stream (the intermediate representation after ANS/prefix decoding, before
// UnpredictICC). Matches libjxl's UnpredictICC (icc_codec.cc).

#ifndef TOOLS_JXLTRAN_COMPRESSED_ICC_H_
#define TOOLS_JXLTRAN_COMPRESSED_ICC_H_

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jxltran {

// Expands |enc| (entropy-decoded ICC payload) into a standard ICC.1 profile.
// Returns false on malformed input; on failure |out_icc| is left unchanged.
bool UnpredictIccProfile(const uint8_t* enc, size_t enc_len,
                         std::vector<uint8_t>* out_icc);
inline bool UnpredictIccProfile(const std::vector<uint8_t>& enc,
                                std::vector<uint8_t>* out_icc) {
  return UnpredictIccProfile(enc.data(), enc.size(), out_icc);
}

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_COMPRESSED_ICC_H_
