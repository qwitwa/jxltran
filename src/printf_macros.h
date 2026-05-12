// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_JXLTRAN_PRINTF_MACROS_H_
#define TOOLS_JXLTRAN_PRINTF_MACROS_H_

// Same rules as lib/jxl/base/printf_macros.h (for tools only).

#if !defined(PRIdS)
#if defined(_WIN64)
#define PRIdS "lld"
#elif defined(_WIN32)
#define PRIdS "d"
#else
#define PRIdS "zd"
#endif
#endif

#if !defined(PRIuS)
#if defined(_WIN64)
#define PRIuS "llu"
#elif defined(_WIN32)
#define PRIuS "u"
#else
#define PRIuS "zu"
#endif
#endif

#endif  // TOOLS_JXLTRAN_PRINTF_MACROS_H_
