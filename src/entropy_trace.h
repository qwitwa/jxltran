// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Optional stderr tracing for entropy / TOC-permutation debugging.
//
//   JXL_ENTROPY_TRACE=1
//       Emit "[jxltran-entropy]" lines (histogram init, context maps, TOC
//       permutation). For libjxl, use the same variable plus
//       JXL_ENTROPY_TRACE_HIST=1 to emit matching ReadHistogram / DecodeHistograms
//       / context-map detail (otherwise libjxl only prints TOC / permutation
//       checkpoints from coeff_order.cc and toc.cc).
//
//   JXL_ENTROPY_TRACE_HIST=1 (optional, jxltran)
//       Reserved for future finer-grained gating; histogram tracing currently
//       follows JXL_ENTROPY_TRACE only.

#pragma once

#include <cstdarg>
#include <cstdio>
#include <cstdlib>

namespace jxltran {

inline bool EntropyTraceEnabled() {
  const char* e = std::getenv("JXL_ENTROPY_TRACE");
  return e != nullptr && e[0] != '\0' && e[0] != '0';
}

inline bool EntropyTraceHist() {
  const char* e = std::getenv("JXL_ENTROPY_TRACE_HIST");
  return e != nullptr && e[0] != '\0' && e[0] != '0';
}

inline void EntropyTraceV(const char* fmt, va_list ap) {
  std::fputs("[jxltran-entropy] ", stderr);
  std::vfprintf(stderr, fmt, ap);
  std::fputc('\n', stderr);
}

inline void EntropyTrace(const char* fmt, ...) {
  if (!EntropyTraceEnabled()) return;
  va_list ap;
  va_start(ap, fmt);
  EntropyTraceV(fmt, ap);
  va_end(ap);
}

}  // namespace jxltran
