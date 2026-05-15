// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "orientation_compose.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <climits>
#include <limits>

namespace jxltran {

namespace {

// kComposeAfter[c-1][d-1] = EXIF tag for (transform delta) ∘ (transform current)
// on stored pixels, matching Pillow's transpose order for EXIF 1..8.
constexpr uint8_t kComposeAfter[8][8] = {
    {1, 2, 3, 4, 5, 6, 7, 8},
    {2, 1, 4, 3, 8, 7, 6, 5},
    {3, 4, 1, 2, 7, 8, 5, 6},
    {4, 3, 2, 1, 6, 5, 8, 7},
    {5, 6, 7, 8, 1, 2, 3, 4},
    {6, 5, 8, 7, 4, 3, 2, 1},
    {7, 8, 5, 6, 3, 4, 1, 2},
    {8, 7, 6, 5, 2, 1, 4, 3},
};

}  // namespace

uint32_t ComposeExifOrientationAfter(uint32_t current, uint32_t delta) {
  return static_cast<uint32_t>(kComposeAfter[current - 1][delta - 1]);
}

void OrientedCanvasDimensions(uint32_t orientation, uint32_t storage_w,
                              uint32_t storage_h, uint32_t* display_w,
                              uint32_t* display_h) {
  if (orientation >= 5 && orientation <= 8) {
    *display_w = storage_h;
    *display_h = storage_w;
  } else {
    *display_w = storage_w;
    *display_h = storage_h;
  }
}

void StoragePixelToDisplay(uint32_t orientation, int64_t storage_x,
                           int64_t storage_y, uint32_t storage_w,
                           uint32_t storage_h, int64_t* display_x,
                           int64_t* display_y) {
  const int64_t sw = static_cast<int64_t>(storage_w);
  const int64_t sh = static_cast<int64_t>(storage_h);
  if (orientation < 1 || orientation > 8) orientation = 1;
  switch (orientation) {
    case 1:
      *display_x = storage_x;
      *display_y = storage_y;
      break;
    case 2:
      *display_x = sw - 1 - storage_x;
      *display_y = storage_y;
      break;
    case 3:
      *display_x = sw - 1 - storage_x;
      *display_y = sh - 1 - storage_y;
      break;
    case 4:
      *display_x = storage_x;
      *display_y = sh - 1 - storage_y;
      break;
    case 5:
      *display_x = storage_y;
      *display_y = storage_x;
      break;
    case 6:
      *display_x = sh - 1 - storage_y;
      *display_y = storage_x;
      break;
    case 7:
      *display_x = sh - 1 - storage_y;
      *display_y = sw - 1 - storage_x;
      break;
    case 8:
      *display_x = storage_y;
      *display_y = sw - 1 - storage_x;
      break;
    default:
      *display_x = storage_x;
      *display_y = storage_y;
      break;
  }
}

void StorageRectToDisplayAabb(uint32_t orientation, uint32_t storage_w,
                              uint32_t storage_h, int64_t sx0, int64_t sy0,
                              uint64_t rect_w, uint64_t rect_h,
                              int64_t* out_x, int64_t* out_y, uint64_t* out_w,
                              uint64_t* out_h) {
  if (rect_w == 0 || rect_h == 0) {
    *out_x = 0;
    *out_y = 0;
    *out_w = 0;
    *out_h = 0;
    return;
  }
  const int64_t x0 = sx0;
  const int64_t y0 = sy0;
  const int64_t x1 = sx0 + static_cast<int64_t>(rect_w) - 1;
  const int64_t y1 = sy0 + static_cast<int64_t>(rect_h) - 1;
  const std::array<std::pair<int64_t, int64_t>, 4> corners = {{
      {x0, y0},
      {x1, y0},
      {x0, y1},
      {x1, y1},
  }};
  int64_t min_dx = std::numeric_limits<int64_t>::max();
  int64_t min_dy = std::numeric_limits<int64_t>::max();
  int64_t max_dx = std::numeric_limits<int64_t>::min();
  int64_t max_dy = std::numeric_limits<int64_t>::min();
  for (const auto& c : corners) {
    int64_t dx = 0;
    int64_t dy = 0;
    StoragePixelToDisplay(orientation, c.first, c.second, storage_w,
                          storage_h, &dx, &dy);
    min_dx = std::min(min_dx, dx);
    min_dy = std::min(min_dy, dy);
    max_dx = std::max(max_dx, dx);
    max_dy = std::max(max_dy, dy);
  }
  *out_x = min_dx;
  *out_y = min_dy;
  *out_w = static_cast<uint64_t>(max_dx - min_dx + 1);
  *out_h = static_cast<uint64_t>(max_dy - min_dy + 1);
}

void DisplayPixelToStorage(uint32_t orientation, int64_t display_x,
                           int64_t display_y, uint32_t storage_w,
                           uint32_t storage_h, int64_t* storage_x,
                           int64_t* storage_y) {
  const int64_t sw = static_cast<int64_t>(storage_w);
  const int64_t sh = static_cast<int64_t>(storage_h);
  if (orientation < 1 || orientation > 8) orientation = 1;
  switch (orientation) {
    case 1:
      *storage_x = display_x;
      *storage_y = display_y;
      break;
    case 2:
      *storage_x = sw - 1 - display_x;
      *storage_y = display_y;
      break;
    case 3:
      *storage_x = sw - 1 - display_x;
      *storage_y = sh - 1 - display_y;
      break;
    case 4:
      *storage_x = display_x;
      *storage_y = sh - 1 - display_y;
      break;
    case 5:
      *storage_x = display_y;
      *storage_y = display_x;
      break;
    case 6:
      *storage_x = display_y;
      *storage_y = sh - 1 - display_x;
      break;
    case 7:
      *storage_x = sw - 1 - display_y;
      *storage_y = sh - 1 - display_x;
      break;
    case 8:
      *storage_x = sw - 1 - display_y;
      *storage_y = display_x;
      break;
    default:
      *storage_x = display_x;
      *storage_y = display_y;
      break;
  }
}

bool ConvertDisplayCropToStorageCanvas(uint32_t orientation,
                                       uint32_t storage_w, uint32_t storage_h,
                                       int32_t display_x, int32_t display_y,
                                       uint32_t display_w, uint32_t display_h,
                                       int32_t* out_x, int32_t* out_y,
                                       uint32_t* out_w, uint32_t* out_h) {
  if (orientation < 1 || orientation > 8) {
    fprintf(stderr,
            "jxltran: internal error: orientation %" PRIu32 " out of range\n",
            orientation);
    return false;
  }
  const int64_t dix = display_x;
  const int64_t diy = display_y;
  const int64_t diw = static_cast<int64_t>(display_w);
  const int64_t dih = static_cast<int64_t>(display_h);
  if (diw <= 0 || dih <= 0) {
    fprintf(stderr, "jxltran: --crop: display width/height must be positive\n");
    return false;
  }
  // Display crop may extend outside the current oriented canvas; the new
  // stored canvas may be larger than existing frame data (implicit padding).
  const int64_t x0 = dix;
  const int64_t y0 = diy;
  const int64_t x1 = dix + diw - 1;
  const int64_t y1 = diy + dih - 1;
  int64_t min_sx = std::numeric_limits<int64_t>::max();
  int64_t min_sy = std::numeric_limits<int64_t>::max();
  int64_t max_sx = std::numeric_limits<int64_t>::min();
  int64_t max_sy = std::numeric_limits<int64_t>::min();
  const std::array<std::pair<int64_t, int64_t>, 4> corners = {{
      {x0, y0},
      {x1, y0},
      {x0, y1},
      {x1, y1},
  }};
  for (const auto& c : corners) {
    int64_t sx = 0;
    int64_t sy = 0;
    DisplayPixelToStorage(orientation, c.first, c.second, storage_w,
                          storage_h, &sx, &sy);
    min_sx = std::min(min_sx, sx);
    min_sy = std::min(min_sy, sy);
    max_sx = std::max(max_sx, sx);
    max_sy = std::max(max_sy, sy);
  }
  if (min_sx > max_sx || min_sy > max_sy) {
    fprintf(stderr, "jxltran: --crop: invalid display rectangle after mapping\n");
    return false;
  }
  const int64_t rw = max_sx - min_sx + 1;
  const int64_t rh = max_sy - min_sy + 1;
  if (min_sx < std::numeric_limits<int32_t>::min() ||
      max_sx > std::numeric_limits<int32_t>::max() ||
      min_sy < std::numeric_limits<int32_t>::min() ||
      max_sy > std::numeric_limits<int32_t>::max()) {
    fprintf(stderr,
            "jxltran: --crop: converted coordinates out of int32 range\n");
    return false;
  }
  if (rw <= 0 || rh <= 0 || rw > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
      rh > static_cast<int64_t>(std::numeric_limits<int32_t>::max()) ||
      rw > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()) ||
      rh > static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
    fprintf(stderr, "jxltran: --crop: converted size out of range\n");
    return false;
  }
  *out_x = static_cast<int32_t>(min_sx);
  *out_y = static_cast<int32_t>(min_sy);
  *out_w = static_cast<uint32_t>(rw);
  *out_h = static_cast<uint32_t>(rh);
  return true;
}

bool DisplayAabbMinToStorageCropOrigin(uint32_t orientation, uint32_t storage_w,
                                      uint32_t storage_h, int64_t display_min_x,
                                      int64_t display_min_y, uint64_t rect_w,
                                      uint64_t rect_h, int32_t* out_sx0,
                                      int32_t* out_sy0) {
  if (orientation < 1 || orientation > 8) {
    fprintf(stderr,
            "jxltran: internal error: orientation %" PRIu32 " out of range\n",
            orientation);
    return false;
  }
  if (rect_w == 0 || rect_h == 0) {
    fprintf(stderr,
            "jxltran: --set-frame-region: frame width/height must be positive\n");
    return false;
  }
  const int64_t sw = static_cast<int64_t>(storage_w);
  const int64_t sh = static_cast<int64_t>(storage_h);
  const int64_t fw = static_cast<int64_t>(rect_w);
  const int64_t fh = static_cast<int64_t>(rect_h);
  int64_t sx0 = 0;
  int64_t sy0 = 0;
  switch (orientation) {
    case 1:
      sx0 = display_min_x;
      sy0 = display_min_y;
      break;
    case 2:
      sx0 = sw - fw - display_min_x;
      sy0 = display_min_y;
      break;
    case 3:
      sx0 = sw - fw - display_min_x;
      sy0 = sh - fh - display_min_y;
      break;
    case 4:
      sx0 = display_min_x;
      sy0 = sh - fh - display_min_y;
      break;
    case 5:
      sy0 = display_min_x;
      sx0 = display_min_y;
      break;
    case 6:
      sy0 = sh - fh - display_min_x;
      sx0 = display_min_y;
      break;
    case 7:
      sy0 = sh - fh - display_min_x;
      sx0 = sw - fw - display_min_y;
      break;
    case 8:
      sy0 = display_min_x;
      sx0 = sw - fw - display_min_y;
      break;
    default:
      return false;
  }
  if (sx0 < INT32_MIN || sx0 > INT32_MAX || sy0 < INT32_MIN ||
      sy0 > INT32_MAX) {
    fprintf(stderr,
            "jxltran: --set-frame-region: converted origin out of int32 range\n");
    return false;
  }
  *out_sx0 = static_cast<int32_t>(sx0);
  *out_sy0 = static_cast<int32_t>(sy0);
  int64_t vdx = 0;
  int64_t vdy = 0;
  uint64_t vdw = 0;
  uint64_t vdh = 0;
  StorageRectToDisplayAabb(orientation, storage_w, storage_h,
                           static_cast<int64_t>(*out_sx0),
                           static_cast<int64_t>(*out_sy0), rect_w, rect_h, &vdx,
                           &vdy, &vdw, &vdh);
  if (vdx != display_min_x || vdy != display_min_y) {
    fprintf(stderr,
            "jxltran: --set-frame-region: display position is not achievable "
            "for this orientation/size (internal check failed)\n");
    return false;
  }
  return true;
}

}  // namespace jxltran
