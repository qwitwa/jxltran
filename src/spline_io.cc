// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "spline_io.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "frame_header.h"
#include "printf_macros.h"

namespace jxltran {

namespace {

bool WantFrame(size_t fi, const std::vector<size_t>* only) {
  if (only == nullptr || only->empty()) {
    return true;
  }
  return std::binary_search(only->begin(), only->end(), fi);
}

void AppendSplineLine(const LfGlobalSplines& spl, size_t spline_index,
                      std::string* out) {
  const QuantizedSplineData& sp = spl.splines[spline_index];
  char buf[64];
  for (size_t c = 0; c < 3; ++c) {
    for (int k = 0; k < 32; ++k) {
      snprintf(buf, sizeof(buf), "%d",
               static_cast<int>(sp.color_dct[c][static_cast<size_t>(k)]));
      *out += buf;
      out->push_back(' ');
    }
  }
  for (int k = 0; k < 32; ++k) {
    snprintf(buf, sizeof(buf), "%d",
             static_cast<int>(sp.sigma_dct[static_cast<size_t>(k)]));
    *out += buf;
    out->push_back(' ');
  }

  const double sx = static_cast<double>(spl.starting_points[spline_index].first);
  const double sy = static_cast<double>(spl.starting_points[spline_index].second);
  snprintf(buf, sizeof(buf), "%.9g %.9g", sx, sy);
  *out += buf;

  // Bitstream stores delta-of-deltas; recover absolute knot positions like
  // libjxl QuantizedSpline::Dequantize (accumulate dd → first differences → xy).
  int64_t ax = static_cast<int64_t>(std::llround(sx));
  int64_t ay = static_cast<int64_t>(std::llround(sy));
  int64_t cur_dx = 0;
  int64_t cur_dy = 0;
  for (const std::pair<int64_t, int64_t>& d : sp.control_points) {
    cur_dx += d.first;
    cur_dy += d.second;
    ax += cur_dx;
    ay += cur_dy;
    snprintf(buf, sizeof(buf), " %.9g %.9g", static_cast<double>(ax),
             static_cast<double>(ay));
    *out += buf;
  }
  out->push_back('\n');
}

const char* SkipWs(const char* p) {
  while (*p == ' ' || *p == '\t' || *p == '\r') {
    ++p;
  }
  return p;
}

bool ParseI32(const char** pp, int32_t* v) {
  *pp = SkipWs(*pp);
  char* end = nullptr;
  const long x = strtol(*pp, &end, 10);
  if (end == *pp) {
    return false;
  }
  *v = static_cast<int32_t>(x);
  *pp = end;
  return true;
}

bool ParseDoubleTok(const char** pp, double* v) {
  *pp = SkipWs(*pp);
  char* end = nullptr;
  *v = strtod(*pp, &end);
  if (end == *pp) {
    return false;
  }
  *pp = end;
  return true;
}

}  // namespace

bool SplinesTextFromCodestream(const ParsedCodestream& cs,
                               const std::vector<size_t>* only_frames,
                               std::string* out) {
  out->clear();
  *out += "# jxltran splines v1\n";
  bool any = false;
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (!WantFrame(fi, only_frames)) {
      continue;
    }
    const FramedUnit& fu = cs.frames[fi];
    if ((fu.frame.flags & kFrameFlagSplines) == 0 || !fu.lf_global_splines) {
      continue;
    }
    const LfGlobalSplines& spl = *fu.lf_global_splines;
    if (spl.splines.empty()) {
      continue;
    }
    char hdr[128];
    snprintf(hdr, sizeof(hdr), "frame %" PRIuS " quant %d\n", fi,
             static_cast<int>(spl.quantization_adjustment));
    *out += hdr;
    for (size_t si = 0; si < spl.splines.size(); ++si) {
      AppendSplineLine(spl, si, out);
    }
    any = true;
  }
  return any;
}

bool SplinesParseText(const std::string& text,
                      std::vector<std::pair<size_t, LfGlobalSplines>>* frames) {
  frames->clear();
  size_t cur_frame = 0;
  bool have_frame = false;
  LfGlobalSplines cur{};

  const char* p = text.c_str();
  std::string line;
  auto flush_frame = [&]() {
    if (!have_frame || cur.splines.empty()) {
      return true;
    }
    if (cur.starting_points.size() != cur.splines.size()) {
      return false;
    }
    frames->emplace_back(cur_frame, std::move(cur));
    cur = LfGlobalSplines{};
    have_frame = false;
    return true;
  };

  while (*p != '\0') {
    const char* eol = std::strchr(p, '\n');
    if (eol == nullptr) {
      line.assign(p);
      p += line.size();
    } else {
      line.assign(p, eol - p);
      p = eol + 1;
    }
    while (!line.empty() &&
           (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
      line.pop_back();
    }
    const char* s = line.c_str();
    s = SkipWs(s);
    if (*s == '\0' || *s == '#') {
      continue;
    }
    if (std::strncmp(s, "frame", 5) == 0 && (s[5] == ' ' || s[5] == '\t')) {
      if (!flush_frame()) {
        return false;
      }
      s += 5;
      s = SkipWs(s);
      char* end = nullptr;
      const unsigned long fi = strtoul(s, &end, 10);
      if (end == s) {
        return false;
      }
      s = SkipWs(end);
      if (std::strncmp(s, "quant", 5) != 0) {
        return false;
      }
      s += 5;
      int32_t q = 0;
      if (!ParseI32(&s, &q)) {
        return false;
      }
      if (*SkipWs(s) != '\0') {
        return false;
      }
      cur_frame = static_cast<size_t>(fi);
      cur.quantization_adjustment = q;
      have_frame = true;
      continue;
    }

    if (!have_frame) {
      return false;
    }

    QuantizedSplineData sp{};
    for (int n = 0; n < 128; ++n) {
      int32_t v = 0;
      if (!ParseI32(&s, &v)) {
        return false;
      }
      if (n < 32) {
        sp.color_dct[0][static_cast<size_t>(n)] = v;
      } else if (n < 64) {
        sp.color_dct[1][static_cast<size_t>(n - 32)] = v;
      } else if (n < 96) {
        sp.color_dct[2][static_cast<size_t>(n - 64)] = v;
      } else {
        sp.sigma_dct[static_cast<size_t>(n - 96)] = v;
      }
    }
    double sx = 0, sy = 0;
    if (!ParseDoubleTok(&s, &sx) || !ParseDoubleTok(&s, &sy)) {
      return false;
    }
    cur.starting_points.emplace_back(static_cast<float>(sx),
                                     static_cast<float>(sy));

    std::vector<std::pair<double, double>> abs_pts;
    abs_pts.emplace_back(sx, sy);
    s = SkipWs(s);
    while (*s != '\0') {
      double px = 0, py = 0;
      if (!ParseDoubleTok(&s, &px) || !ParseDoubleTok(&s, &py)) {
        return false;
      }
      abs_pts.emplace_back(px, py);
      s = SkipWs(s);
    }
    if (abs_pts.size() < 2) {
      return false;
    }
    // Text lists absolute knot positions; the codestream uses delta-of-deltas
    // (libjxl QuantizedSpline::Create): dd[0] = P1-P0, dd[j] = Δ[j]-Δ[j-1].
    const size_t nseg = abs_pts.size() - 1;
    std::vector<std::pair<int64_t, int64_t>> deltas;
    deltas.reserve(nseg);
    for (size_t i = 1; i < abs_pts.size(); ++i) {
      const int64_t px =
          static_cast<int64_t>(std::llround(abs_pts[i - 1].first));
      const int64_t py =
          static_cast<int64_t>(std::llround(abs_pts[i - 1].second));
      const int64_t nx =
          static_cast<int64_t>(std::llround(abs_pts[i].first));
      const int64_t ny =
          static_cast<int64_t>(std::llround(abs_pts[i].second));
      deltas.emplace_back(nx - px, ny - py);
    }
    sp.control_points.resize(nseg);
    sp.control_points[0] = deltas[0];
    for (size_t j = 1; j < nseg; ++j) {
      sp.control_points[j].first = deltas[j].first - deltas[j - 1].first;
      sp.control_points[j].second = deltas[j].second - deltas[j - 1].second;
    }
    cur.splines.push_back(std::move(sp));
  }

  if (!flush_frame()) {
    return false;
  }
  if (frames->empty()) {
    return false;
  }
  return true;
}

}  // namespace jxltran
