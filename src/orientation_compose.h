// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef TOOLS_JXLTRAN_SRC_ORIENTATION_COMPOSE_H_
#define TOOLS_JXLTRAN_SRC_ORIENTATION_COMPOSE_H_

#include <cstdint>

namespace jxltran {

// Composes EXIF-style orientation tags (1..8, JEITA / libjxl semantics) as
// transforms applied to stored pixels: first |current|, then |delta|.
// Returns the single tag that describes the combined transform.
// |current| and |delta| must each be in [1, 8].
uint32_t ComposeExifOrientationAfter(uint32_t current, uint32_t delta);

// Display size (as djxl / JxlDecoder with default orientation handling) for
// a stored canvas of width |storage_w| and height |storage_h|.
void OrientedCanvasDimensions(uint32_t orientation, uint32_t storage_w,
                                uint32_t storage_h, uint32_t* display_w,
                                uint32_t* display_h);

// Inverse of DisplayPixelToStorage: maps a storage-canvas pixel (codestream
// coordinates) to display (djxl-oriented) pixel coordinates for EXIF tag
// |orientation| in [1,8].
void StoragePixelToDisplay(uint32_t orientation, int64_t storage_x,
                           int64_t storage_y, uint32_t storage_w,
                           uint32_t storage_h, int64_t* display_x,
                           int64_t* display_y);

// Axis-aligned bounding box in display space of the inclusive storage
// rectangle [sx0, sy0] .. [sx0 + rect_w - 1, sy0 + rect_h - 1].
void StorageRectToDisplayAabb(uint32_t orientation, uint32_t storage_w,
                              uint32_t storage_h, int64_t sx0, int64_t sy0,
                              uint64_t rect_w, uint64_t rect_h,
                              int64_t* out_x, int64_t* out_y, uint64_t* out_w,
                              uint64_t* out_h);

// Maps a display pixel (top-left origin, as after UndoOrientation in
// lib/jxl/dec_external_image.cc) to stored codestream canvas coordinates.
void DisplayPixelToStorage(uint32_t orientation, int64_t display_x,
                           int64_t display_y, uint32_t storage_w,
                           uint32_t storage_h, int64_t* storage_x,
                           int64_t* storage_y);

// Converts a crop rectangle given in display coordinates (WxH+X+Y from the
// user) into the equivalent stored-canvas rectangle for ApplyCrop. |orientation|
// is the file's EXIF tag before any --set-orientation in this run. There is no
// clamping to the previous stored canvas: the new canvas may extend past
// existing frame samples (implicit zero / transparent padding). Returns false
// and prints to stderr if the tag is invalid, dimensions are non-positive,
// mapped bounds are inconsistent, coordinates overflow int32/uint32, or
// converted sizes overflow.
bool ConvertDisplayCropToStorageCanvas(uint32_t orientation,
                                       uint32_t storage_w, uint32_t storage_h,
                                       int32_t display_x, int32_t display_y,
                                       uint32_t display_w, uint32_t display_h,
                                       int32_t* out_x, int32_t* out_y,
                                       uint32_t* out_w, uint32_t* out_h);

// Inverse of StorageRectToDisplayAabb for a fixed-size axis-aligned storage
// rectangle (rect_w × rect_h): find the storage origin (sx0, sy0) whose
// display-space AABB top-left is (display_min_x, display_min_y). Returns false
// if orientation is invalid, rect size is zero, or the target is not exactly
// realizable (sanity check).
bool DisplayAabbMinToStorageCropOrigin(uint32_t orientation, uint32_t storage_w,
                                      uint32_t storage_h, int64_t display_min_x,
                                      int64_t display_min_y, uint64_t rect_w,
                                      uint64_t rect_h, int32_t* out_sx0,
                                      int32_t* out_sy0);

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_SRC_ORIENTATION_COMPOSE_H_
