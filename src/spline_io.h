// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_JXLTRAN_SPLINE_IO_H_
#define TOOLS_JXLTRAN_SPLINE_IO_H_

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "codestream.h"
#include "lfglobal.h"

namespace jxltran {

// Append one `frame N quant Q` section and one text line per spline (see
// README / --help). |only_frames| null = all frames with splines.
bool SplinesTextFromCodestream(const ParsedCodestream& cs,
                               const std::vector<size_t>* only_frames,
                               std::string* out);

// Parse text produced by SplinesTextFromCodestream. Each entry is
// (codestream_frame_index, splines bundle for that frame).
bool SplinesParseText(const std::string& text,
                      std::vector<std::pair<size_t, LfGlobalSplines>>* frames);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_SPLINE_IO_H_
