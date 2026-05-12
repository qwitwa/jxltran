// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_JXLTRAN_TRACE_H_
#define TOOLS_JXLTRAN_TRACE_H_

#include <cstddef>
#include <cstdio>

namespace jxltran {

// Enable (--verbose) or disable trace output. |out| is typically stderr.
void TraceInit(FILE* out_or_null);
bool TraceIsOn();

// Base bit offset in the codestream for the current BitReader (0 for the main
// codestream reader; frame_start_byte*8 before parsing a frame).
void TraceSetReadBase(size_t codestream_base_bit);

// Log a parsed field. |rel_bit| is the bit index at field start, relative to
// the BitReader buffer; the printed position is (TraceSetReadBase + rel_bit)
// split as BYTE+BIT (bit = position mod 8).
void TraceReadField(size_t rel_bit, const char* field, const char* fmt, ...);

// Log an emitted field. |abs_bit| is the absolute bit index from the start of
// the output codestream (BitWriter::bit_pos() before writing the field).
void TraceWriteField(size_t abs_bit, const char* field, const char* fmt, ...);

// Marker line at an absolute codestream bit (frame boundaries, splices).
void TraceMarkerAt(size_t abs_bit, const char* field, const char* fmt, ...);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_TRACE_H_
