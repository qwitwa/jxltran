// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Standalone ISOBMFF box parser and serializer for JXL containers.
// No dependencies outside the C++ standard library.

#ifndef TOOLS_JXLTRAN_BOX_H_
#define TOOLS_JXLTRAN_BOX_H_

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace jxltran {

// A parsed ISOBMFF box. |data| contains the box payload (after the
// 8/16-byte box header), not including the type or size fields.
struct JxlBox {
  char type[4];
  std::vector<uint8_t> data;
};

// Parses all top-level ISOBMFF boxes from |buf|[0..|size|).
// Returns false and prints a message to stderr on malformed input.
bool ParseBoxes(const uint8_t* buf, size_t size, std::vector<JxlBox>* out);

// Appends the serialized box (size + type + data) to |out|.
void SerializeBox(const JxlBox& box, std::vector<uint8_t>* out);

// Writes the pre-formatted 12-byte JXL signature box to |out|.
void WriteJxlSignatureBox(std::vector<uint8_t>* out);

// Writes a 20-byte JXL ftyp box (brand "jxl ", compatible "jxl ") to |out|.
void WriteJxlFtypBox(std::vector<uint8_t>* out);

// Returns true if |buf| starts with the JXL container magic bytes.
bool IsJxlContainer(const uint8_t* buf, size_t size);

// Returns true if |buf| starts with the JXL codestream magic bytes (0xFF 0x0A).
bool IsJxlCodestream(const uint8_t* buf, size_t size);

// Collects jxlc and jxlp boxes from |boxes|, sorts jxlp boxes by their
// counter value (low 31 bits of the 4-byte jxlp header word), strips the
// 4-byte headers from jxlp payloads, and concatenates everything into
// |codestream|. Returns false on error (e.g. duplicate counters, mixed
// jxlc+jxlp, missing boxes).
bool ReassembleCodestream(const std::vector<JxlBox>& boxes,
                          std::vector<uint8_t>* codestream);

// Returns true if |boxes| contains any jxlp box.
bool HasJxlpBoxes(const std::vector<JxlBox>& boxes);

// Returns true if |boxes| contains any box that is not a structural container
// box (i.e. not JXL , ftyp, jxll, jxlc, or jxlp). Such boxes require the
// container format to be preserved.
bool HasMetadataBoxes(const std::vector<JxlBox>& boxes);

// Reorders jxlp boxes in |boxes| so they appear in ascending counter order.
// Non-jxlp boxes that precede the first codestream box are kept as a prefix;
// any boxes that appear after the last codestream box become a suffix.
// Returns false on error (duplicate counters, mixed jxlc+jxlp).
bool SortJxlpBoxes(std::vector<JxlBox>* boxes);

// Replaces all jxlp (or jxlc) boxes in |boxes| with a single jxlc box
// containing the reassembled codestream. Non-codestream boxes are preserved
// in their original positions relative to the codestream.
// Returns false on error.
bool MergeJxlpBoxes(std::vector<JxlBox>* boxes);

// Flags for StripBoxesByType. Combinable with bitwise OR.
static constexpr uint32_t kStripExif = 1 << 0;   // "Exif" boxes
static constexpr uint32_t kStripXmp = 1 << 1;    // "xml " boxes
static constexpr uint32_t kStripJumbf = 1 << 2;  // "jumb" boxes
static constexpr uint32_t kStripJbrd = 1 << 3;   // "jbrd" boxes
// kStripAll removes every box that is not structural (JXL , ftyp, jxll,
// jxlc, jxlp). This includes unknown/custom box types.
static constexpr uint32_t kStripAll = ~0u;

// Removes boxes whose type matches |strip_flags| from |boxes|.
// brob boxes whose wrapped type (first 4 payload bytes) matches are also
// removed. Structural container boxes (JXL , ftyp, jxll, jxlc, jxlp) are
// never removed regardless of flags.
void StripBoxesByType(std::vector<JxlBox>* boxes, uint32_t strip_flags);

// Controls where non-structural metadata boxes are placed relative to the
// codestream when reordering.
enum class BoxOrder {
  kAsIs,              // preserve the input box ordering (default)
  kBeforeCodestream,  // all metadata precedes the codestream
  kAfterCodestream,   // all metadata follows the codestream
  kExplicit,          // explicit comma-separated type list; unspecified boxes
              // are appended at the end in their original relative order
};

// Reorders boxes according to |order|. Structural prefix boxes (JXL , ftyp,
// jxll) always stay first. kAsIs is a no-op.
// For kExplicit, |explicit_order| gives the desired sequence of box types;
// boxes not listed are appended at the end preserving their original order.
void ReorderBoxes(std::vector<JxlBox>* boxes, BoxOrder order,
                  const std::vector<std::array<char, 4>>& explicit_order = {});

// ── brob (brotli compression) support ─────────────────────────────────────
// Only available when compiled with JXLTRAN_HAVE_BROTLI.

// Controls brob (brotli-compressed box) handling.
enum class BrobOpt {
  kAsIs,       // preserve brob boxes unchanged (default)
  kCompress,   // compress eligible metadata boxes into brob format
  kDecompress  // decompress brob boxes back to their original form
};

#ifdef JXLTRAN_HAVE_BROTLI
// Decompresses all brob boxes in |boxes| to their original form.
// Returns false if any brotli decompression fails.
bool DecompressBrobBoxes(std::vector<JxlBox>* boxes);

// Compresses eligible metadata boxes that are not already in brob format.
// If |types| is non-empty, only boxes whose type is in that list are
// compressed; if empty, all eligible types (Exif, xml , jumb) are compressed.
// Uses maximum brotli quality. Returns false if any compression fails.
bool CompressMetadataBoxes(std::vector<JxlBox>* boxes,
                           const std::vector<std::array<char, 4>>& types);
#endif

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_BOX_H_
