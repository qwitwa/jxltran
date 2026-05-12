// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "trace.h"

#include <cstdarg>
#include <cstdio>

#include "printf_macros.h"

namespace jxltran {
namespace {

FILE* g_out = nullptr;
size_t g_read_base = 0;

void Emit(size_t abs_bit, const char* field, const char* fmt, va_list ap) {
  if (!g_out) return;
  const size_t by = abs_bit / 8u;
  const unsigned bi = static_cast<unsigned>(abs_bit & 7u);
  fprintf(g_out, "%" PRIuS "+%u | %s | ", by, bi, field);
  vfprintf(g_out, fmt, ap);
  fputc('\n', g_out);
}

}  // namespace

void TraceInit(FILE* out_or_null) {
  g_out = out_or_null;
  g_read_base = 0;
}

bool TraceIsOn() { return g_out != nullptr; }

void TraceSetReadBase(size_t codestream_base_bit) {
  g_read_base = codestream_base_bit;
}

void TraceReadField(size_t rel_bit, const char* field, const char* fmt, ...) {
  if (!g_out) return;
  va_list ap;
  va_start(ap, fmt);
  Emit(g_read_base + rel_bit, field, fmt, ap);
  va_end(ap);
}

void TraceWriteField(size_t abs_bit, const char* field, const char* fmt, ...) {
  if (!g_out) return;
  va_list ap;
  va_start(ap, fmt);
  Emit(abs_bit, field, fmt, ap);
  va_end(ap);
}

void TraceMarkerAt(size_t abs_bit, const char* field, const char* fmt, ...) {
  if (!g_out) return;
  va_list ap;
  va_start(ap, fmt);
  Emit(abs_bit, field, fmt, ap);
  va_end(ap);
}

}  // namespace jxltran
