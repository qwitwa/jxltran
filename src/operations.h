// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Higher-level codestream operations: mutate a ParsedCodestream in memory.

#ifndef TOOLS_JXLTRAN_OPERATIONS_H_
#define TOOLS_JXLTRAN_OPERATIONS_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "codestream.h"

namespace jxltran {

struct GabArgs {
  enum class Kind { kNone, kBlur, kSharpen, kCustom };
  Kind kind = Kind::kNone;
  // kBlur / kSharpen: A>=0; moves each XYB channel in logit(w1+w2) with w1+w2
  // clamped to [-0.45, 2]. Same A on blur then sharpen (or the reverse)
  // cancels. Sharpen A=1 from the default sum moves halfway toward the min sum
  // (max sharp). Weights near implicit encoder defaults snap to gab_custom off.
  float amount = 0.f;
  // kCustom: gab_x_weight1, gab_x_weight2, gab_y_*, gab_b_* (unnormalized).
  float custom[6] = {};
};

struct HeaderMod {
  uint32_t set_orientation = 0;
  uint32_t set_bits_per_sample = 0;
  bool have_set_num_loops = false;
  uint32_t set_num_loops = 0;
  bool have_set_tps = false;
  uint32_t set_tps_numerator = 1;
  uint32_t set_tps_denominator = 1;
};

// Applies header-only modifications to |cs| (image header fields).
bool ApplyHeaderMod(ParsedCodestream* cs, const HeaderMod& mod);

// Reversible metadata-only crop / canvas resize. (dx,dy) are relative to the
// current canvas: output pixel (px,py) shows input (px+dx, py+dy). New canvas
// size is (new_w x new_h). Encoded frame dimensions are unchanged; only
// canvas size and Regular / SkipProgressive frame origins (and have_crop when
// needed) are updated. LF and ReferenceOnly frame headers are not modified.
bool ApplyCrop(ParsedCodestream* cs, int32_t dx, int32_t dy, uint32_t new_w,
               uint32_t new_h);

// Repositions codestream frame |frame_index|'s display rectangle to
// (display_x, display_y) with the same display width/height as before
// (WxH+X+Y in the same oriented display space as `jxltran --info` and --crop).
// LF, reference-only, and unknown frame types are rejected; a frame without
// have_crop (full canvas) cannot be moved. |display_space_orientation| must
// be the image header orientation before any --set-orientation in this run
// (same basis as --crop). Intended to run before ApplyCrop. If |out_changed|
// is non-null, sets *out_changed when the frame header was modified.
bool ApplySetFrameRegionFromDisplay(ParsedCodestream* cs, size_t frame_index,
                                    uint32_t display_space_orientation,
                                    int32_t display_x, int32_t display_y,
                                    uint32_t display_w, uint32_t display_h,
                                    bool* out_changed = nullptr);

// Reversible Gaborish (restoration filter) weight edits on the codestream
// headers. LF and reference-only frames are skipped. Decoded pixels change
// when the effective kernel changes.
// If |only_frames| is non-null, only codestream frames at those indices are
// considered (LF / reference-only are still skipped); null means all frames.
bool ApplyGabArgs(ParsedCodestream* cs, const GabArgs& args,
                  const std::vector<size_t>* only_frames = nullptr);

// Decoder-effective Gaborish weights (gab off → zeros; implicit default gab →
// libjxl defaults; packed all_default restoration → decoder defaults, not the
// Gaborish-off expansion used before applying blur/sharpen in ApplyGabArgs).
// Order: gab_x_weight1, gab_x_weight2, gab_y_weight1, gab_y_weight2,
// gab_b_weight1, gab_b_weight2.
void GabEffectiveWeights6(bool modular_encoding, bool restoration_all_default,
                          const ParsedRestorationFilter& rf, float out6[6]);

// Set EPF iteration count (0–3) in the restoration filter bundle. LF and
// reference-only frames are skipped. Packed all_default restoration is
// expanded to explicit libjxl defaults first (like other frame-tail edits).
bool ApplyEpfIters(ParsedCodestream* cs, uint32_t epf_iters,
                   const std::vector<size_t>* only_frames = nullptr);

// Scale overall EPF strength by |factor| (>1 stronger, <1 weaker). VarDCT
// multiplies the effective epf_quant_mul (implicit 0.46 when epf_sigma_custom
// is false); modular multiplies epf_sigma_for_modular (implicit 1.0 when EPF
// is enabled). Values within tolerance of decoder defaults snap back like
// implicit defaults (epf_sigma_custom off for VarDCT; exact 1.0 sigma for
// modular). LF / reference-only frames and frames with epf_iters==0 are
// skipped. |factor| must be finite and >0; factor==1 is a no-op.
bool ApplyEpfAmplitudeScale(ParsedCodestream* cs, float factor,
                            const std::vector<size_t>* only_frames = nullptr);

// For --info: when EPF iterations > 0, sets |*out_amp| to the effective amplitude
// (VarDCT: epf_quant_mul or implicit 0.46; modular: epf_sigma_for_modular).
bool EpfEffectiveAmplitudeForInfo(bool modular_encoding,
                                  bool restoration_all_default,
                                  const ParsedRestorationFilter& rf,
                                  float* out_amp);

// VarDCT + EPF: effective EPF sharp LUT "uniformity" in [0,1] (0 = implicit
// decoder ramp i/7, 1 = all ones; see --set-epf-uniformity). 0 when sharp LUT
// is not custom.
float EpfSharpUniformityForInfo(bool restoration_all_default,
                                const ParsedRestorationFilter& rf);

// VarDCT only (skipped on modular, LF, reference-only, and when epf_iters==0).
// |uniformity| in [0,1]: mixes decoder-default ramp {0,1/7,...,1} with all-1.
// 0 clears epf_sharp_custom (implicit ramp). |out_changed| set when any frame
// bundle actually changed (optional).
bool ApplyEpfSharpUniformity(ParsedCodestream* cs, float uniformity,
                             const std::vector<size_t>* only_frames = nullptr,
                             bool* out_changed = nullptr);

// Photon noise: |change| false = leave bitstream noise as-is. |iso| == 0
// clears kNoise and removes the 80-bit LUT from DC global (when present).
// |iso| > 0 sets kNoise and LUT from the same ISO model as libjxl
// (SimulatePhotonNoise). Applies to modular and VarDCT regular /
// skip-progressive frames. DC-global layout (patches / splines before the LUT)
// is parsed at ReadCodestream time; if that parse fails while patches or
// splines are enabled, --set-photon-noise-iso errors. When a TOC permutation is
// present, DC-global size changes keep the permutation prefix verbatim and only
// re-encode U32 section sizes (no Lehmer re-encode).
// If |only_frames| is non-null, only those codestream frame indices are
// modified; null means all eligible frames.
bool ApplyPhotonNoiseIso(ParsedCodestream* cs, bool change, float iso,
                         const std::vector<size_t>* only_frames = nullptr);

// Replace or insert LF-global spline entropy from a text file (see
// --extract-splines). Requires a successful LF-global prefix parse at read
// time. If the frame had no kSplines flag, the flag is set and a new bundle is
// inserted before the noise LUT (or equivalent). The re-encoded spline bundle
// may use any bit length; the frame body stays byte-aligned by appending zero
// padding bits after the last payload bit when needed (bit phase after the
// spline region may change). Padding is derived from bit counts only, so
// trailing zero bits already present as encoder padding in the last body byte
// cannot be distinguished from payload—at worst roughly one redundant byte
// may be emitted versus a full LF-global decode (out of scope). Incompatible
// with --set-photon-noise-iso on the same frame or TOC strip on the same pass.
bool ApplySplinesFromFile(ParsedCodestream* cs, const char* path,
                          const std::vector<size_t>* only_frames = nullptr);

// Set UTF-8 frame name bytes (|name_utf8|) on regular / skip-progressive
// frames. Indices are codestream indices after any other transforms applied
// earlier in the same jxltran invocation (e.g. after --keep-listed-frames).
// Empty |name_utf8| clears the name.
bool ApplySetFrameNames(
    ParsedCodestream* cs,
    const std::vector<std::pair<size_t, std::vector<uint8_t>>>&
        frame_index_name);

// Keep only the listed codestream frames, in the order given (first mention
// wins; duplicate indices are skipped). Verbatim frame bytes are
// preserved; |is_last| is fixed on the last kept frame. The last kept frame
// must be regular or skip-progressive (not LF-only / reference-only tail).
// After reorder, each non-last frame's |save_as_reference| is set to the
// primary |blending_info.source| of the following frame when that field is
// serialized (so reference slots line up with the new decode order).
// Returns false if an index is out of range or the tail constraint fails.
// |out_changed| is set when the frame list or any |is_last| / save_as_reference
// field changed relative to a full identity keep 0..n-1 (optional).
bool ApplyKeepListedFrames(ParsedCodestream* cs,
                           const std::vector<size_t>& frames_in_order,
                           bool* out_changed = nullptr);

// Blend override for one codestream frame (regular / skip-progressive only).
// |mode| is always applied (0–4 or spec slug at CLI). Optional fields are applied
// only when the corresponding |set_*| flag is true (after mode normalization).
struct FrameBlendOverride {
  size_t frame_index = 0;
  uint32_t mode = 0;
  bool set_alpha_channel = false;
  uint32_t alpha_channel = 0;
  bool set_clamp = false;
  bool clamp = false;
  bool set_source = false;
  uint32_t source = 0;
  bool set_save_as_reference = false;
  uint32_t save_as_reference = 0;
};

// Optional fields are applied as given when set_* is true (no range checks
// against num_extra or mode; invalid combinations may confuse decoders).
// target= sets save_as_reference (0–3) only when the frame is not last.
//
// |frame_index| values are codestream indices before --keep-listed-frames.
// When both keep and blend overrides are used in one run, jxltran applies
// keep first and remaps overrides via RemapFrameBlendOverridesForKeepOrder so
// ApplyKeepListedFrames does not clobber blend edits.
bool RemapFrameBlendOverridesForKeepOrder(
    const std::vector<size_t>& frames_in_order, size_t num_frames_before,
    std::vector<FrameBlendOverride>* overrides);
bool ApplyFrameBlendOverrides(
    ParsedCodestream* cs,
    const std::vector<FrameBlendOverride>& overrides);

// |frame_index_duration|: (codestream frame index, duration in ticks). Only
// when the image has animation; only regular / skip-progressive frames.
bool ApplyFrameDurationOverrides(
    ParsedCodestream* cs,
    const std::vector<std::pair<size_t, uint32_t>>& frame_index_duration);

// Best-effort concatenation of two bare codestreams: keeps the primary image
// header (canvas becomes max(width)×max(height) of the two), clears is_last on
// the primary's last frame, appends all frames from the second file (skipping
// the second image header in the byte stream). On each side, regular /
// skip-progressive frames that use implicit full-canvas framing get an explicit
// crop to the source file canvas size when the merged canvas is larger in either
// dimension. Header compatibility is only: same num_extra; both XYB or both
// non-XYB; when non-XYB, same main BitDepth — unless |skip_header_compat| is
// true (then this check is skipped). On failure |*err| is a short English
// message (no trailing newline).
bool AppendCodestreamMerge(const std::vector<uint8_t>& primary_cs,
                            const std::vector<uint8_t>& append_cs,
                            std::vector<uint8_t>* merged_out, std::string* err,
                            bool skip_header_compat = false);

// Fixed 42×3 modular zero-frame bare codestream (16 bytes; base64
// "/woQAFIASAgGAQAMAEsgGA=="). After AppendCodestreamMerge the former primary
// tail is forced to save_as_reference=1 and the appended tail is kAdd with
// blending source 1 so the zero patch is added to that buffer (true no-op).
std::vector<uint8_t> BuiltinAppendDummyTailCodestream();

}  // namespace jxltran

#endif  // TOOLS_JXLTRAN_OPERATIONS_H_
