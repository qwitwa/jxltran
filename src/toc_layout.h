// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verbose human-readable description of TOC → frame section layout (stream
// order vs logical indices, byte offsets in the codestream).

#ifndef TOOLS_JXLTRAN_TOC_LAYOUT_H_
#define TOOLS_JXLTRAN_TOC_LAYOUT_H_

#include <cstdio>

namespace jxltran {

struct ParsedCodestream;

// Prints one line per TOC slot in **stream order** (concatenated frame body):
// codestream byte offset, physical index, size, logical id, and a section
// label (lf_global / lf_group / hf_global / hf_pass…). When a TOC
// permutation is present, |logical| is decoded from the permutation (matches
// libjxl ReadToc + FrameDecoder::InitFrame).
void PrintCodestreamTocLayoutVerbose(FILE* out, const ParsedCodestream& cs);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_TOC_LAYOUT_H_
