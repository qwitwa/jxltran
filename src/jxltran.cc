// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// jxltran: lossless JPEG XL bitstream remuxing tool.
// Similar to jpegtran but for JXL: rewrites the container without
// decode+encode, preserving the codestream unchanged.

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "box.h"
#include "cmdline.h"
#include "codestream.h"
#include "image_header.h"
#include "operations.h"
#include "orientation_compose.h"
#include "printf_macros.h"
#include "trace.h"

namespace {

// CLI value meaning "leave unchanged". Legacy alias "as-is" is still accepted.
static bool ParseKeepToken(const char* str) {
  return strcmp(str, "keep") == 0 || strcmp(str, "as-is") == 0;
}

bool ReadFile(const char* path, std::vector<uint8_t>* out) {
  FILE* f = fopen(path, "rb");
  if (!f) {
    fprintf(stderr, "Failed to open '%s' for reading\n", path);
    return false;
  }
  fseek(f, 0, SEEK_END);
  long size = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (size < 0) {
    fclose(f);
    return false;
  }
  out->resize(static_cast<size_t>(size));
  if (fread(out->data(), 1, out->size(), f) != out->size()) {
    fprintf(stderr, "Failed to read '%s'\n", path);
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

bool WriteFile(const char* path, const std::vector<uint8_t>& data) {
  FILE* f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "Failed to open '%s' for writing\n", path);
    return false;
  }
  if (fwrite(data.data(), 1, data.size(), f) != data.size()) {
    fprintf(stderr, "Failed to write '%s'\n", path);
    fclose(f);
    return false;
  }
  fclose(f);
  return true;
}

enum class ContainerOpt {
  kAsIs,  // keep the same format as the input
  kNo,   // bare codestream: strip container and metadata (see --container help)
  kYes,  // always wrap in a container
  kIfNeeded  // use container only when metadata boxes are present
};

enum class JxlpOpt {
  kAsIs,  // keep jxlp boxes exactly as in the input
  kSort,  // reorder jxlp boxes by counter (in-order delivery, separate boxes)
  kMerge  // sort + merge all jxlp/jxlc into a single jxlc box
};

static bool ParseContainerOpt(const char* str, ContainerOpt* out) {
  if (ParseKeepToken(str)) {
    *out = ContainerOpt::kAsIs;
    return true;
  }
  if (strcmp(str, "no") == 0) {
    *out = ContainerOpt::kNo;
    return true;
  }
  if (strcmp(str, "yes") == 0) {
    *out = ContainerOpt::kYes;
    return true;
  }
  if (strcmp(str, "if-needed") == 0) {
    *out = ContainerOpt::kIfNeeded;
    return true;
  }
  fprintf(
      stderr,
      "Unknown --container value '%s'. Expected: keep, no, yes, if-needed\n",
      str);
  return false;
}

static bool ParseJxlpOpt(const char* str, JxlpOpt* out) {
  if (ParseKeepToken(str)) {
    *out = JxlpOpt::kAsIs;
    return true;
  }
  if (strcmp(str, "sort") == 0) {
    *out = JxlpOpt::kSort;
    return true;
  }
  if (strcmp(str, "merge") == 0) {
    *out = JxlpOpt::kMerge;
    return true;
  }
  fprintf(stderr, "Unknown --jxlp value '%s'. Expected: keep, sort, merge\n",
          str);
  return false;
}

// Parse a comma-separated list of strip types into a bitmask.
// Accepted tokens: exif, xmp, jumbf, jbrd, all.
static bool ParseStripOpt(const char* str, uint32_t* out) {
  *out = 0;
  const char* p = str;
  while (p && *p) {
    const char* end = strchr(p, ',');
    size_t len = end ? static_cast<size_t>(end - p) : strlen(p);
    if (strncmp(p, "exif", len) == 0 && len == 4) {
      *out |= jxltran::kStripExif;
    } else if (strncmp(p, "xmp", len) == 0 && len == 3) {
      *out |= jxltran::kStripXmp;
    } else if (strncmp(p, "jumbf", len) == 0 && len == 5) {
      *out |= jxltran::kStripJumbf;
    } else if (strncmp(p, "jbrd", len) == 0 && len == 4) {
      *out |= jxltran::kStripJbrd;
    } else if (strncmp(p, "all", len) == 0 && len == 3) {
      *out = jxltran::kStripAll;
      return true;
    } else {
      fprintf(stderr,
              "Unknown --strip type '%.*s'. "
              "Expected: exif, xmp, jumbf, jbrd, all\n",
              static_cast<int>(len), p);
      return false;
    }
    p = end ? end + 1 : nullptr;
  }
  return true;
}

// Maps a friendly box-type name (or a raw 4-char type) to a 4-byte box type.
static bool ParseBoxType(const char* p, size_t len, std::array<char, 4>* out) {
  static const struct {
    const char* friendly;
    const char* type4;
  } kMap[] = {
      {"exif", "Exif"}, {"xmp", "xml "},  {"jumbf", "jumb"}, {"jumb", "jumb"},
      {"jbrd", "jbrd"}, {"jxlc", "jxlc"}, {"jxlp", "jxlp"},  {"jxli", "jxli"},
      {"jxll", "jxll"}, {"brob", "brob"}, {"ftyp", "ftyp"},  {nullptr, nullptr},
  };
  for (int i = 0; kMap[i].friendly; ++i) {
    if (strlen(kMap[i].friendly) == len &&
        strncmp(p, kMap[i].friendly, len) == 0) {
      memcpy(out->data(), kMap[i].type4, 4);
      return true;
    }
  }
  if (len == 4) {  // raw 4-char type
    memcpy(out->data(), p, 4);
    return true;
  }
  fprintf(stderr, "Unknown box type '%.*s'\n", static_cast<int>(len), p);
  return false;
}

static bool ParseBoxTypeList(const char* str,
                             std::vector<std::array<char, 4>>* out) {
  const char* p = str;
  while (p && *p) {
    const char* end = strchr(p, ',');
    size_t len = end ? static_cast<size_t>(end - p) : strlen(p);
    std::array<char, 4> t;
    if (!ParseBoxType(p, len, &t)) return false;
    out->push_back(t);
    p = end ? end + 1 : nullptr;
  }
  return true;
}

// Holds a box-order mode plus, for kExplicit, the ordered type list.
struct BoxOrderArg {
  jxltran::BoxOrder order = jxltran::BoxOrder::kAsIs;
  std::vector<std::array<char, 4>> types;
};

static bool ParseBoxOrderOpt(const char* str, BoxOrderArg* out) {
  if (ParseKeepToken(str)) {
    out->order = jxltran::BoxOrder::kAsIs;
    return true;
  }
  if (strcmp(str, "before-codestream") == 0) {
    out->order = jxltran::BoxOrder::kBeforeCodestream;
    return true;
  }
  if (strcmp(str, "after-codestream") == 0) {
    out->order = jxltran::BoxOrder::kAfterCodestream;
    return true;
  }
  // Otherwise treat as a comma-separated explicit type list.
  out->order = jxltran::BoxOrder::kExplicit;
  return ParseBoxTypeList(str, &out->types);
}

// Holds a brob mode plus, for kCompress, the types to compress (empty = all).
struct BrobArg {
  jxltran::BrobOpt mode = jxltran::BrobOpt::kAsIs;
  std::vector<std::array<char, 4>> types;  // empty means all eligible types
};

static bool ParseBrobOpt(const char* str, BrobArg* out) {
  if (ParseKeepToken(str)) {
    out->mode = jxltran::BrobOpt::kAsIs;
    return true;
  }
  if (strcmp(str, "decompress") == 0) {
    out->mode = jxltran::BrobOpt::kDecompress;
    return true;
  }
  if (strcmp(str, "compress") == 0) {
    out->mode = jxltran::BrobOpt::kCompress;
    return true;
  }
  if (strncmp(str, "compress:", 9) == 0) {
    out->mode = jxltran::BrobOpt::kCompress;
    return ParseBoxTypeList(str + 9, &out->types);
  }
  fprintf(stderr,
          "Unknown --brob value '%s'. "
          "Expected: keep, compress, compress:TYPE[,...], decompress\n",
          str);
  return false;
}
static bool ParseOrientationOpt(const char* s, uint32_t* out) {
  char* end;
  long v = strtol(s, &end, 10);
  if (*end != '\0') {
    fprintf(stderr,
            "orientation value: expected [1,8] or 90, 180, or 270 (degrees "
            "clockwise), got '%s'\n",
            s);
    return false;
  }
  if (v >= 1 && v <= 8) {
    *out = static_cast<uint32_t>(v);
    return true;
  }
  // Clockwise rotation aliases (EXIF tags 6, 3, 8).
  if (v == 90) {
    *out = 6;
    return true;
  }
  if (v == 180) {
    *out = 3;
    return true;
  }
  if (v == 270) {
    *out = 8;
    return true;
  }
  fprintf(stderr,
          "orientation value: expected [1,8] or 90, 180, or 270 (degrees "
          "clockwise), got '%s'\n",
          s);
  return false;
}

static bool ParseBitsPerSampleOpt(const char* s, uint32_t* out) {
  char* end;
  long v = strtol(s, &end, 10);
  if (*end != '\0' || v < 1 || v > 32) {
    fprintf(stderr,
            "--set-bits-per-sample: expected a value in [1,32], got '%s'\n", s);
    return false;
  }
  *out = static_cast<uint32_t>(v);
  return true;
}

struct NumLoopsArg {
  bool set = false;
  uint32_t value = 0;
};

static bool ParseNumLoopsOpt(const char* s, NumLoopsArg* out) {
  char* end;
  long v = strtol(s, &end, 10);
  if (*end != '\0' || v < 0) {
    fprintf(stderr,
            "--set-num-loops: expected a non-negative integer, got '%s'\n", s);
    return false;
  }
  out->set = true;
  out->value = static_cast<uint32_t>(v);
  return true;
}

struct TpsArg {
  bool set = false;
  // If true, |percent_value| is resolved against the file's current TPS after
  // ReadCodestream; then cleared and |numerator|/|denominator| are filled.
  bool relative_percent = false;
  double percent_value = 100.0;
  uint32_t numerator = 1;
  uint32_t denominator = 1;
};

static bool ParseTpsOpt(const char* s, TpsArg* out) {
  size_t len = strlen(s);
  while (len > 0 && std::isspace(static_cast<unsigned char>(s[len - 1]))) {
    --len;
  }
  if (len == 0) {
    fprintf(stderr, "--set-tps: empty value\n");
    return false;
  }
  if (s[len - 1] == '%') {
    const char* const pct = s + len - 1;
    char* end = nullptr;
    const double value = strtod(s, &end);
    while (end < pct && std::isspace(static_cast<unsigned char>(*end))) {
      ++end;
    }
    if (end != pct || !(value > 0.0)) {
      fprintf(stderr,
              "--set-tps: expected a positive percentage (e.g. 50%%), got "
              "'%s'\n",
              s);
      return false;
    }
    out->set = true;
    out->relative_percent = true;
    out->percent_value = value;
    out->numerator = 1;
    out->denominator = 1;
    return true;
  }

  char* slash = const_cast<char*>(strchr(s, '/'));
  char* end;
  long num = strtol(s, &end, 10);
  if (num <= 0 || (end != slash && *end != '\0')) {
    fprintf(stderr,
            "--set-tps: expected a positive percentage (e.g. 50%%), a positive "
            "integer N, or N/D, got '%s'\n",
            s);
    return false;
  }
  long den = 1;
  if (slash) {
    den = strtol(slash + 1, &end, 10);
    if (den <= 0 || *end != '\0') {
      fprintf(stderr,
              "--set-tps: expected a positive percentage (e.g. 50%%), a "
              "positive integer N, or N/D, got "
              "'%s'\n",
              s);
      return false;
    }
  }
  out->set = true;
  out->relative_percent = false;
  out->numerator = static_cast<uint32_t>(num);
  out->denominator = static_cast<uint32_t>(den);
  return true;
}

static bool ResolveTpsPercentFromMetadata(const jxltran::ImageMetadata& meta,
                                          TpsArg* tps) {
  if (!tps->relative_percent) return true;
  if (!meta.have_animation) {
    fprintf(stderr,
            "jxltran: --set-tps: percentage values require an animated image "
            "(have_animation=true)\n");
    return false;
  }
  const uint32_t on = meta.animation.tps_numerator;
  const uint32_t od = meta.animation.tps_denominator;
  if (on == 0 || od == 0) {
    fprintf(stderr,
            "jxltran: --set-tps: ticks per second in file is zero/invalid\n");
    return false;
  }
  const double cur = static_cast<double>(on) / static_cast<double>(od);
  const double next = cur * (tps->percent_value / 100.0);
  if (!(next > 0.0) || !std::isfinite(next)) {
    fprintf(stderr, "jxltran: --set-tps: invalid scaled TPS\n");
    return false;
  }
  constexpr uint64_t kScale = 1000000;
  uint64_t n = static_cast<uint64_t>(std::llround(next * kScale));
  uint64_t d = kScale;
  if (n == 0) n = 1;
  uint64_t g = std::gcd(n, d);
  n /= g;
  d /= g;
  if (n > std::numeric_limits<uint32_t>::max() ||
      d > std::numeric_limits<uint32_t>::max()) {
    fprintf(
        stderr,
        "jxltran: --set-tps: scaled TPS exceeds uint32 range; try a smaller "
        "percentage or use N/D\n");
    return false;
  }
  tps->relative_percent = false;
  tps->numerator = static_cast<uint32_t>(n);
  tps->denominator = static_cast<uint32_t>(d);
  return true;
}

// WxH or WxH followed by two signed offsets (e.g. 3000x2000+500+500,
// 4000x3000-500-500). Output pixel (px,py) shows input (px+X, py+Y).
struct CropArg {
  bool set = false;
  int32_t x = 0, y = 0;
  uint32_t w = 0, h = 0;
};

static bool ParseCropOpt(const char* s, CropArg* out) {
  unsigned wu = 0, hu = 0;
  int consumed = 0;
  if (sscanf(s, "%ux%u%n", &wu, &hu, &consumed) != 2 || wu == 0 || hu == 0) {
    fprintf(stderr,
            "--crop: expected WxH or WxH with two signed offsets "
            "(e.g. 3000x2000+500+500), got '%s'\n",
            s);
    return false;
  }
  const char* tail = s + consumed;
  int32_t x = 0, y = 0;
  if (*tail != '\0') {
    char* endx = nullptr;
    long xl = strtol(tail, &endx, 10);
    if (endx == tail) {
      fprintf(stderr, "--crop: invalid X offset in '%s'\n", s);
      return false;
    }
    char* endy = nullptr;
    long yl = strtol(endx, &endy, 10);
    if (endy == endx) {
      fprintf(stderr,
              "--crop: expected two signed offsets after WxH (e.g. "
              "3000x2000+500+500), got '%s'\n",
              s);
      return false;
    }
    if (*endy != '\0') {
      fprintf(stderr, "--crop: trailing garbage after offsets in '%s'\n", s);
      return false;
    }
    if (xl < INT32_MIN || xl > INT32_MAX || yl < INT32_MIN || yl > INT32_MAX) {
      fprintf(stderr, "--crop: offsets out of range in '%s'\n", s);
      return false;
    }
    x = static_cast<int32_t>(xl);
    y = static_cast<int32_t>(yl);
  }
  out->set = true;
  out->w = wu;
  out->h = hu;
  out->x = x;
  out->y = y;
  return true;
}

static void PatchFtypMinorVersion(std::vector<jxltran::JxlBox>* boxes,
                                  uint32_t version) {
  for (auto& box : *boxes) {
    if (memcmp(box.type, "ftyp", 4) == 0 && box.data.size() >= 8) {
      box.data[4] = (version >> 24) & 0xFF;
      box.data[5] = (version >> 16) & 0xFF;
      box.data[6] = (version >> 8) & 0xFF;
      box.data[7] = version & 0xFF;
      break;
    }
  }
}

static bool ParseGabBlur(const char* s, jxltran::GabArgs* out) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0' || v < 0.f) {
    fprintf(stderr,
            "jxltran: --gab-blur expects a non-negative float, got '%s'\n", s);
    return false;
  }
  out->kind = jxltran::GabArgs::Kind::kBlur;
  out->amount = v;
  return true;
}

static bool ParseGabSharpen(const char* s, jxltran::GabArgs* out) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0' || v < 0.f) {
    fprintf(stderr,
            "jxltran: --gab-sharpen expects a non-negative float, got '%s'\n",
            s);
    return false;
  }
  out->kind = jxltran::GabArgs::Kind::kSharpen;
  out->amount = v;
  return true;
}

static bool ParseGabWeights(const char* s, jxltran::GabArgs* out) {
  float c[6];
  const int n = std::sscanf(s, "%f,%f,%f,%f,%f,%f", &c[0], &c[1], &c[2], &c[3],
                            &c[4], &c[5]);
  if (n != 6) {
    fprintf(stderr,
            "jxltran: --gab-weights expects six comma-separated floats "
            "(gab_x_weight1,2, gab_y_weight1,2, gab_b_weight1,2), got '%s'\n",
            s);
    return false;
  }
  out->kind = jxltran::GabArgs::Kind::kCustom;
  std::memcpy(out->custom, c, sizeof(c));
  return true;
}

static bool StoreGabStringArg(const char* s, const char** out) {
  *out = s;
  return true;
}

struct PhotonIsoArg {
  bool set = false;
  float iso = 0.f;
};

static bool ParsePhotonIsoOpt(const char* s, PhotonIsoArg* out) {
  char* end = nullptr;
  const float v = strtof(s, &end);
  if (end == s || *end != '\0' || v < 0.f || !std::isfinite(v)) {
    fprintf(stderr,
            "jxltran: --set-photon-noise-iso expects a non-negative finite "
            "float (0 = remove synthetic noise, >0 = nominal ISO), got '%s'\n",
            s);
    return false;
  }
  out->set = true;
  out->iso = v;
  return true;
}

struct Args {
  const char* file_in = nullptr;
  const char* file_out = nullptr;
  bool info = false;
  ContainerOpt container = ContainerOpt::kAsIs;
  JxlpOpt jxlp = JxlpOpt::kAsIs;
  uint32_t strip = 0;
  BoxOrderArg box_order;
  BrobArg brob;
  uint32_t set_orientation = 0;      // 0 = no change; 1-8 = new orientation
  uint32_t rel_orientation = 0;    // 0 = no change; 1-8 = compose after current
  uint32_t set_bits_per_sample = 0;  // 0 = no change; only for xyb_encoded
  NumLoopsArg num_loops;
  TpsArg tps;
  CropArg crop;
  PhotonIsoArg photon_iso;
  const char* gab_blur = nullptr;
  const char* gab_sharpen = nullptr;
  const char* gab_weights = nullptr;
  bool verbose = false;

  int GabOptionCount() const {
    return (gab_blur != nullptr) + (gab_sharpen != nullptr) +
           (gab_weights != nullptr);
  }

  bool NeedsHeaderMod() const {
    return set_orientation != 0 || rel_orientation != 0 ||
           set_bits_per_sample != 0 || num_loops.set || tps.set;
  }

  bool NeedsPhotonNoiseIso() const { return photon_iso.set; }

  void AddCommandLineOptions(jpegxl::tools::CommandLineParser* cmdline) {
    cmdline->AddPositionalOption("INPUT", /*required=*/true,
                                 "The JPEG XL input file.", &file_in);
    cmdline->AddPositionalOption(
        "OUTPUT", /*required=*/false,
        "The JPEG XL output file (omit with --info).", &file_out);
    cmdline->AddHelpText("\nOutput options:", 0);
    cmdline->AddOptionValue(
        '\0', "container", "keep|no|yes|if-needed",
        "Container output mode (default: keep).\n"
        "    keep      : preserve input format (container or bare "
        "codestream).\n"
        "    no        : output only the bare codestream. Drops the ISOBMFF\n"
        "                wrapper and all metadata boxes (Exif, XMP, JUMBF, "
        "…).\n"
        "                Multiple jxlp payloads are concatenated in counter\n"
        "                order, producing the same codestream bytes as after\n"
        "                --jxlp merge (merge is implied; --jxlp is ignored "
        "for\n"
        "                bare output).\n"
        "    yes       : always wrap in a container.\n"
        "    if-needed : use container only when metadata boxes are present.",
        &container, ParseContainerOpt);
    cmdline->AddOptionValue(
        '\0', "jxlp", "keep|sort|merge",
        "jxlp box handling when the output is a container (default: keep).\n"
        "    keep  : keep jxlp boxes unchanged in the written container.\n"
        "    sort  : reorder jxlp boxes by counter (normalize OOO delivery).\n"
        "    merge : sort and merge all jxlp into a single jxlc box.",
        &jxlp, ParseJxlpOpt);
    cmdline->AddOptionValue(
        '\0', "strip", "exif|xmp|jumbf|jbrd|all[,...]",
        "Strip metadata boxes from the container (comma-separated list).\n"
        "    exif  : strip Exif boxes (and brob wrappers around them).\n"
        "    xmp   : strip XMP boxes (xml  type).\n"
        "    jumbf : strip JUMBF boxes.\n"
        "    jbrd  : strip JPEG bitstream reconstruction data.\n"
        "    all   : strip all non-structural metadata.",
        &strip, ParseStripOpt);
    cmdline->AddOptionValue(
        '\0', "box-order",
        "keep|before-codestream|after-codestream|TYPE[,TYPE...]",
        "Position of boxes in the output container (default: keep).\n"
        "    keep               : preserve input box order.\n"
        "    before-codestream  : all metadata precedes the codestream.\n"
        "    after-codestream   : all metadata follows the codestream.\n"
        "    TYPE[,TYPE...]     : explicit order (e.g. exif,jbrd,jxlc,xmp);\n"
        "                         unspecified boxes are appended at the end.\n"
        "    Known aliases: exif, xmp, jumbf, jbrd, jxlc, jxlp, jxli, jxll,\n"
        "                   brob, ftyp; or any raw 4-char box type.",
        &box_order, ParseBoxOrderOpt);
    cmdline->AddOptionValue(
        '\0', "brob", "keep|compress[:TYPE[,...]]|decompress",
        "brob (brotli-compressed box) handling (default: keep)."
#ifdef JXLTRAN_HAVE_BROTLI
        "\n"
        "    keep               : preserve brob boxes unchanged.\n"
        "    compress           : compress all eligible metadata\n"
        "                         (Exif/XMP/JUMBF) into brob format.\n"
        "    compress:TYPE[,...]: compress only the listed types,\n"
        "                         e.g. compress:xmp or compress:xmp,jumbf.\n"
        "    decompress         : decompress brob boxes to their original "
        "form.",
#else
        "\n    (brotli support not compiled in; only keep is effective)",
#endif
        &brob, ParseBrobOpt);
    cmdline->AddHelpText("\nImage header options:", 0);
    cmdline->AddOptionValue('\0', "set-orientation", "1..8|90|180|270",
                            "Set the orientation field in the image header.\n"
                            "Orientation values (Exif semantics):\n"
                            "  1 = normal (no transform)\n"
                            "  2 = flip horizontally\n"
                            "  3 = rotate 180 (alias: 180)\n"
                            "  4 = flip vertically\n"
                            "  5 = transpose (rotate 90 CW + flip H)\n"
                            "  6 = rotate 90 clockwise (alias: 90)\n"
                            "  7 = flip H + rotate 90 CW\n"
                            "  8 = rotate 90 counter-clockwise (alias: 270)\n"
                            "Aliases 90, 180, 270 are degrees of rotation clockwise.",
                            &set_orientation, ParseOrientationOpt);
    cmdline->AddOptionValue(
        '\0', "set-orientation-relative", "1..8|90|180|270",
        "Like --set-orientation, but |delta| is composed *after* the file's\n"
        "current orientation tag (Exif semantics). Example: if the file is\n"
        "90° CCW (8) and you pass --set-orientation-relative=6 (90° CW), the\n"
        "result is identity (1). Value 1 leaves the orientation tag unchanged.\n"
        "Rotation aliases 90, 180, 270 are clockwise degrees (same as\n"
        "--set-orientation). Mutually exclusive with --set-orientation.",
        &rel_orientation, ParseOrientationOpt);
    cmdline->AddOptionValue(
        '\0', "set-bits-per-sample", "1..32",
        "Set the bits_per_sample field in the image header.\n"
        "Only valid for XYB-encoded images (xyb_encoded=true), where the\n"
        "bit depth is metadata hinting the decoder about desired output\n"
        "precision; it does not affect the compressed image data.",
        &set_bits_per_sample, ParseBitsPerSampleOpt);
    cmdline->AddOptionValue(
        '\0', "set-num-loops", "N",
        "Set the animation loop count (0 = loop forever).\n"
        "Only valid for animated images (have_animation=true).",
        &num_loops, ParseNumLoopsOpt);
    cmdline->AddOptionValue(
        '\0', "set-tps", "N[/D]|P%",
        "Set the animation ticks-per-second.\n"
        "  Absolute: N or N/D (positive integers), e.g. 24, 25, 30, 60, or\n"
        "            30000/1001 (NTSC).\n"
        "  Relative: P%% scales the file's current TPS (from the image "
        "header),\n"
        "            e.g. 50%% for half speed, 200%% for double.\n"
        "Only valid for animated images (have_animation=true).",
        &tps, ParseTpsOpt);
    cmdline->AddHelpText("\nCrop options:", 0);
    cmdline->AddOptionValue(
        '\0', "crop", "WxH or WxHXY",
        "Reversible metadata-only canvas resize / crop. WxH is the new canvas\n"
        "size. X and Y are optional signed offsets: output pixel (px,py) "
        "shows\n"
        "input pixel (px+X, py+Y). Coordinates are in display space (as "
        "decoders\n"
        "such as djxl show the image after applying the orientation tag); "
        "when\n"
        "the tag is identity, this matches codestream pixel coordinates. "
        "Any\n"
        "signed combination is allowed (e.g. 500x500+200-300 on a 600x600 "
        "image\n"
        "can add padding on the right/bottom). Encoded frame dimensions stay "
        "as in\n"
        "the bitstream; when a frame has no crop, it is treated as "
        "full-canvas for\n"
        "positioning. LF and ReferenceOnly frames are not modified. No "
        "compressed\n"
        "sample data is re-encoded.",
        &crop, ParseCropOpt);
    cmdline->AddHelpText("\nPhoton noise (VarDCT) options:", 0);
    cmdline->AddOptionValue(
        '\0', "set-photon-noise-iso", "ISO",
        "Rewrite the synthetic noise LUT in each regular / skip-progressive "
        "frame\n"
        "using the same ISO-based model as libjxl (cjxl --photon_noise_iso). "
        "0\n"
        "clears the kNoise flag and removes the 80-bit LUT from DC global. "
        "Requires\n"
        "frames without patches or splines and, when the DC-global byte size "
        "changes,\n"
        "a TOC without permutation. Compressed sample values are unchanged.",
        &photon_iso, ParsePhotonIsoOpt);
    cmdline->AddHelpText("\nGaborish (restoration filter) options:", 0);
    cmdline->AddOptionValue(
        '\0', "gab-blur", "A",
        "Stronger reversible Gaborish blur than the encoder default (A>=0).\n"
        "Uses the same unnormalized gab_x/y/b_weight1 and weight2 on all "
        "channels,\n"
        "scaled from libjxl defaults. Requires explicit frame headers; LF and\n"
        "reference-only frames are unchanged. Mutually exclusive with "
        "--gab-sharpen and --gab-weights.",
        &gab_blur, StoreGabStringArg);
    cmdline->AddOptionValue(
        '\0', "gab-sharpen", "A",
        "Sharpening by scaling default neighbor weights (A>=0). Larger A "
        "pushes\n"
        "gab_*_weight1/2 lower and can make them slightly negative for "
        "stronger\n"
        "sharpening; libjxl rejects only near-singular kernels where\n"
        "|1+4*(weight1+weight2)| < 1e-8 per channel. Mutually exclusive with\n"
        "--gab-blur and --gab-weights.",
        &gab_sharpen, StoreGabStringArg);
    cmdline->AddOptionValue(
        '\0', "gab-weights", "x1,x2,y1,y2,b1,b2",
        "Set all six Gaborish weights (comma-separated, unnormalized).\n"
        "Order: gab_x_weight1, gab_x_weight2, gab_y_weight1, gab_y_weight2,\n"
        "gab_b_weight1, gab_b_weight2. Values may be negative if each channel\n"
        "still satisfies |1+4*(w1+w2)| >= 1e-8 after normalization. Mutually\n"
        "exclusive with --gab-blur and --gab-sharpen.",
        &gab_weights, StoreGabStringArg);
    cmdline->AddHelpText("\nDebugging:", 0);
    cmdline->AddOptionFlag(
        'v', "verbose",
        "Log codestream header parse and rewrite to stderr.\n"
        "Each line: BYTE+BIT | field | value. Positions are absolute from the "
        "start of the codestream (reads) or from the start of the written "
        "codestream (writes), split as byte offset + bit index within that "
        "byte.",
        &verbose, &jpegxl::tools::SetBooleanTrue);
    cmdline->AddOptionFlag(
        '\0', "info",
        "Print oriented display dimensions, animation flag, and one "
        "frame_region line per frame (WxH+X+Y in display space, same "
        "orientation as dimensions), then exit. No OUTPUT file; cannot be "
        "combined with transform or remux options.",
        &info, &jpegxl::tools::SetBooleanTrue);
  }
};

}  // namespace

static bool InfoModeUsesNonDefaultOptions(const Args& a) {
  return a.container != ContainerOpt::kAsIs || a.jxlp != JxlpOpt::kAsIs ||
         a.strip != 0 ||
         a.box_order.order != jxltran::BoxOrder::kAsIs ||
         a.brob.mode != jxltran::BrobOpt::kAsIs || a.set_orientation != 0 ||
         a.rel_orientation != 0 || a.set_bits_per_sample != 0 ||
         a.num_loops.set || a.tps.set || a.crop.set || a.photon_iso.set ||
         a.gab_blur != nullptr || a.gab_sharpen != nullptr ||
         a.gab_weights != nullptr;
}

static void PrintJxlInfoStdout(const jxltran::ParsedCodestream& parsed) {
  uint32_t orient = parsed.image.metadata.orientation;
  if (orient < 1 || orient > 8) orient = 1;
  uint32_t dw = 0;
  uint32_t dh = 0;
  const uint32_t canvas_w = parsed.image.size.width;
  const uint32_t canvas_h = parsed.image.size.height;
  jxltran::OrientedCanvasDimensions(orient, canvas_w, canvas_h, &dw, &dh);
  printf("dimensions: %u x %u\n", dw, dh);
  printf("animation: %s\n",
         parsed.image.metadata.have_animation ? "yes" : "no");

  for (size_t i = 0; i < parsed.frames.size(); ++i) {
    const jxltran::FrameHeader& fh = parsed.frames[i].frame;
    int64_t sx0 = 0;
    int64_t sy0 = 0;
    uint64_t rw = canvas_w;
    uint64_t rh = canvas_h;
    if (fh.have_crop) {
      const int32_t x0 = (fh.frame_type != jxltran::kFrameTypeReferenceOnly)
                             ? jxltran::UnpackSigned(fh.ux0)
                             : 0;
      const int32_t y0 = (fh.frame_type != jxltran::kFrameTypeReferenceOnly)
                             ? jxltran::UnpackSigned(fh.uy0)
                             : 0;
      sx0 = x0;
      sy0 = y0;
      rw = fh.crop_width;
      rh = fh.crop_height;
    }
    int64_t dx = 0;
    int64_t dy = 0;
    uint64_t dtw = 0;
    uint64_t dth = 0;
    jxltran::StorageRectToDisplayAabb(orient, canvas_w, canvas_h, sx0, sy0, rw,
                                        rh, &dx, &dy, &dtw, &dth);
    printf("frame_region: %" PRIuS " %" PRIu64 "x%" PRIu64, i, dtw, dth);
    if (dx >= 0) {
      printf("+%" PRId64, dx);
    } else {
      printf("%" PRId64, dx);
    }
    if (dy >= 0) {
      printf("+%" PRId64 "\n", dy);
    } else {
      printf("%" PRId64 "\n", dy);
    }
  }
}

int main(int argc, const char* argv[]) {
  Args args;
  jpegxl::tools::CommandLineParser cmdline;
  args.AddCommandLineOptions(&cmdline);

  if (!cmdline.Parse(argc, argv)) {
    fprintf(stderr, "Use '%s -h' for more information.\n", argv[0]);
    return 1;
  }
  if (cmdline.HelpFlagPassed() || !args.file_in) {
    cmdline.PrintHelp();
    return 0;
  }
  if (args.GabOptionCount() > 1) {
    fprintf(stderr,
            "jxltran: use at most one of --gab-blur, --gab-sharpen, "
            "--gab-weights\n");
    return 1;
  }
  if (args.set_orientation != 0 && args.rel_orientation != 0) {
    fprintf(stderr,
            "jxltran: use only one of --set-orientation and "
            "--set-orientation-relative\n");
    return 1;
  }
  if (args.info && args.file_out) {
    fprintf(stderr,
            "jxltran: --info does not take an output file; omit OUTPUT.\n");
    return 1;
  }
  if (args.info && InfoModeUsesNonDefaultOptions(args)) {
    fprintf(stderr,
            "jxltran: --info cannot be combined with other options (only "
            "optional -v/--verbose).\n");
    return 1;
  }
  if (!args.info && !args.file_out) {
    fprintf(stderr, "No output file specified.\n");
    return 1;
  }
  jxltran::GabArgs gab_args{};
  if (args.gab_blur) {
    if (!ParseGabBlur(args.gab_blur, &gab_args)) return 1;
  } else if (args.gab_sharpen) {
    if (!ParseGabSharpen(args.gab_sharpen, &gab_args)) return 1;
  } else if (args.gab_weights) {
    if (!ParseGabWeights(args.gab_weights, &gab_args)) return 1;
  }

  jxltran::TraceInit(args.verbose ? stderr : nullptr);

  std::vector<uint8_t> input;
  if (!ReadFile(args.file_in, &input)) return 1;

  bool is_container = jxltran::IsJxlContainer(input.data(), input.size());
  bool is_codestream = jxltran::IsJxlCodestream(input.data(), input.size());

  if (!is_container && !is_codestream) {
    fprintf(stderr, "Input file is not a JPEG XL file.\n");
    return 1;
  }

  // Parse boxes if input is a container; otherwise |codestream| holds the raw
  // bytes and |boxes| stays empty.
  std::vector<jxltran::JxlBox> boxes;
  std::vector<uint8_t> codestream;
  if (is_container) {
    if (!jxltran::ParseBoxes(input.data(), input.size(), &boxes)) return 1;
  } else {
    codestream = input;
  }

  if (args.info) {
    std::vector<uint8_t> cs;
    if (is_container) {
      if (!jxltran::ReassembleCodestream(boxes, &cs)) return 1;
    } else {
      cs = std::move(codestream);
    }
    jxltran::ParsedCodestream parsed;
    if (!jxltran::ReadCodestream(cs.data(), cs.size(), &parsed)) return 1;
    PrintJxlInfoStdout(parsed);
    return 0;
  }

  // ── Apply strip ───────────────────────────────────────────────────────────
  if (is_container && args.strip != 0) {
    jxltran::StripBoxesByType(&boxes, args.strip);
  }

  // ── Apply brob transform ──────────────────────────────────────────────────
  if (is_container && args.brob.mode != jxltran::BrobOpt::kAsIs) {
#ifdef JXLTRAN_HAVE_BROTLI
    if (args.brob.mode == jxltran::BrobOpt::kDecompress) {
      if (!jxltran::DecompressBrobBoxes(&boxes)) return 1;
    } else {
      if (!jxltran::CompressMetadataBoxes(&boxes, args.brob.types)) return 1;
    }
#else
    fprintf(stderr,
            "jxltran was compiled without brotli support; "
            "--brob=%s is not available.\n",
            args.brob.mode == jxltran::BrobOpt::kCompress ? "compress"
                                                          : "decompress");
    return 1;
#endif
  }

  // ── Apply jxlp transform ─────────────────────────────────────────────────
  if (is_container && args.jxlp != JxlpOpt::kAsIs) {
    if (args.jxlp == JxlpOpt::kSort) {
      if (!jxltran::SortJxlpBoxes(&boxes)) return 1;
      PatchFtypMinorVersion(&boxes, 0);
    } else {  // kMerge
      if (!jxltran::MergeJxlpBoxes(&boxes)) return 1;
      PatchFtypMinorVersion(&boxes, 0);
    }
  }

  // ── Apply box ordering ────────────────────────────────────────────────────
  if (is_container && args.box_order.order != jxltran::BoxOrder::kAsIs) {
    jxltran::ReorderBoxes(&boxes, args.box_order.order, args.box_order.types);
  }

  // ── Codestream transforms (read → mutate ParsedCodestream → write)
  // ───────────
  const bool needs_codestream_transform =
      args.NeedsHeaderMod() || args.crop.set || args.GabOptionCount() > 0 ||
      args.NeedsPhotonNoiseIso();
  if (needs_codestream_transform) {
    auto transform_codestream = [&](std::vector<uint8_t>* cs) -> bool {
      const uint8_t* orig_bytes = cs->data();
      const size_t orig_size = cs->size();
      jxltran::ParsedCodestream parsed;
      if (!jxltran::ReadCodestream(orig_bytes, orig_size, &parsed)) {
        return false;
      }
      if (args.tps.set &&
          !ResolveTpsPercentFromMetadata(parsed.image.metadata, &args.tps)) {
        return false;
      }
      uint32_t input_orient_for_crop = parsed.image.metadata.orientation;
      if (input_orient_for_crop < 1 || input_orient_for_crop > 8) {
        input_orient_for_crop = 1;
      }
      const uint32_t input_sw_for_crop = parsed.image.size.width;
      const uint32_t input_sh_for_crop = parsed.image.size.height;
      if (args.NeedsHeaderMod()) {
        jxltran::HeaderMod mod;
        if (args.set_orientation != 0) {
          mod.set_orientation = args.set_orientation;
        } else if (args.rel_orientation != 0) {
          uint32_t cur = parsed.image.metadata.orientation;
          if (cur < 1 || cur > 8) cur = 1;
          mod.set_orientation = jxltran::ComposeExifOrientationAfter(
              cur, args.rel_orientation);
        } else {
          mod.set_orientation = 0;
        }
        mod.set_bits_per_sample = args.set_bits_per_sample;
        mod.have_set_num_loops = args.num_loops.set;
        mod.set_num_loops = args.num_loops.value;
        mod.have_set_tps = args.tps.set;
        mod.set_tps_numerator = args.tps.numerator;
        mod.set_tps_denominator = args.tps.denominator;
        if (!jxltran::ApplyHeaderMod(&parsed, mod)) return false;
      }
      bool wrote_anything = args.NeedsHeaderMod();
      if (args.crop.set) {
        int32_t crop_x = args.crop.x;
        int32_t crop_y = args.crop.y;
        uint32_t crop_w = args.crop.w;
        uint32_t crop_h = args.crop.h;
        if (input_orient_for_crop != 1) {
          if (!jxltran::ConvertDisplayCropToStorageCanvas(
                  input_orient_for_crop, input_sw_for_crop, input_sh_for_crop,
                  args.crop.x, args.crop.y, args.crop.w, args.crop.h, &crop_x,
                  &crop_y, &crop_w, &crop_h)) {
            return false;
          }
        }
        const bool crop_noop = crop_w == input_sw_for_crop &&
                               crop_h == input_sh_for_crop && crop_x == 0 &&
                               crop_y == 0;
        if (!crop_noop) {
          if (!jxltran::ApplyCrop(&parsed, crop_x, crop_y, crop_w, crop_h)) {
            return false;
          }
          wrote_anything = true;
        }
      }
      if (args.GabOptionCount() > 0) {
        if (!jxltran::ApplyGabArgs(&parsed, gab_args)) return false;
        wrote_anything = true;
      }
      if (args.NeedsPhotonNoiseIso()) {
        if (!jxltran::ApplyPhotonNoiseIso(&parsed, true, args.photon_iso.iso)) {
          return false;
        }
        wrote_anything = true;
      }
      if (!wrote_anything) {
        return true;
      }
      std::vector<uint8_t> out;
      if (!jxltran::WriteCodestream(parsed, orig_bytes, &out)) return false;
      *cs = std::move(out);
      return true;
    };

    auto replace_codestream_boxes = [&](std::vector<uint8_t> cs) {
      boxes.erase(std::remove_if(boxes.begin(), boxes.end(),
                                 [](const jxltran::JxlBox& b) {
                                   return memcmp(b.type, "jxlc", 4) == 0 ||
                                          memcmp(b.type, "jxlp", 4) == 0;
                                 }),
                  boxes.end());
      size_t insert_at = 0;
      for (size_t i = 0; i < boxes.size(); ++i) {
        if (memcmp(boxes[i].type, "JXL ", 4) == 0 ||
            memcmp(boxes[i].type, "ftyp", 4) == 0 ||
            memcmp(boxes[i].type, "jxll", 4) == 0) {
          insert_at = i + 1;
        }
      }
      jxltran::JxlBox jxlc;
      memcpy(jxlc.type, "jxlc", 4);
      jxlc.data = std::move(cs);
      boxes.insert(boxes.begin() + insert_at, std::move(jxlc));
      PatchFtypMinorVersion(&boxes, 0);
    };

    if (is_container) {
      std::vector<uint8_t> cs;
      if (!jxltran::ReassembleCodestream(boxes, &cs)) return 1;
      if (!transform_codestream(&cs)) return 1;
      replace_codestream_boxes(std::move(cs));
    } else {
      if (!transform_codestream(&codestream)) return 1;
    }
  }

  // ── Determine whether output uses a container ────────────────────────────
  bool output_container;
  switch (args.container) {
    case ContainerOpt::kAsIs:
      output_container = is_container;
      break;
    case ContainerOpt::kNo:
      output_container = false;
      break;
    case ContainerOpt::kYes:
      output_container = true;
      break;
    case ContainerOpt::kIfNeeded:
      output_container = is_container && jxltran::HasMetadataBoxes(boxes);
      break;
  }

  // ── Produce output ────────────────────────────────────────────────────────
  std::vector<uint8_t> output;
  if (!output_container) {
    // Output a bare codestream.
    if (is_container) {
      if (!jxltran::ReassembleCodestream(boxes, &output)) return 1;
    } else {
      output = std::move(codestream);
    }
  } else {
    // Output a container.
    if (is_container) {
      // Serialize the (possibly transformed) box list.
      for (const auto& box : boxes) {
        jxltran::SerializeBox(box, &output);
      }
    } else {
      // Wrap bare codestream in a minimal container.
      jxltran::WriteJxlSignatureBox(&output);
      jxltran::WriteJxlFtypBox(&output);
      jxltran::JxlBox jxlc;
      memcpy(jxlc.type, "jxlc", 4);
      jxlc.data = std::move(codestream);
      jxltran::SerializeBox(jxlc, &output);
    }
  }

  if (!WriteFile(args.file_out, output)) return 1;
  return 0;
}
