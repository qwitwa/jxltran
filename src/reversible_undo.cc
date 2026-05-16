// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "reversible_undo.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

#include "bits.h"
#include "codestream.h"
#include "frame_header.h"
#include "operations.h"
#include "orientation_compose.h"
#include "opsin_adjust.h"
#include "printf_macros.h"
#include "restoration_filter.h"

namespace jxltran {

namespace {

std::string ShellQuotePath(const char* s) {
  std::string out;
  out.push_back('\'');
  for (const char* p = s; *p; ++p) {
    if (*p == '\'') {
      out += "'\\''";
    } else {
      out.push_back(*p);
    }
  }
  out.push_back('\'');
  return out;
}

std::string HexLowerFromUtf8(const std::vector<uint8_t>& bytes) {
  static const char kHex[] = "0123456789abcdef";
  std::string h;
  h.reserve(bytes.size() * 2);
  for (uint8_t b : bytes) {
    h.push_back(kHex[b >> 4]);
    h.push_back(kHex[b & 15]);
  }
  return h;
}

// Even-length all-hex-ASCII names would parse as hex if emitted literally
// (e.g. "6809" vs UTF-8 for U+6809); force hex for those.
bool NameBytesAmbiguousEvenHexAscii(const std::vector<uint8_t>& name) {
  if (name.empty() || (name.size() % 2) != 0) return false;
  for (uint8_t b : name) {
    if (!std::isxdigit(static_cast<unsigned char>(b))) return false;
  }
  return true;
}

// Printable ASCII (33–126) safe as an unquoted INDEX:NAME token; comma/colon
// break pair parsing; space requires "" (see AppendFrameNameTokenForUndo).
bool NameBytesEmitAsUndoUnquotedLiteral(const std::vector<uint8_t>& name) {
  if (name.empty()) return false;
  for (uint8_t b : name) {
    if (b < 33 || b > 126) return false;
    if (b == ',' || b == ':') return false;
  }
  return !NameBytesAmbiguousEvenHexAscii(name);
}

bool NameBytesEmitAsUndoQuotedAscii(const std::vector<uint8_t>& name) {
  if (name.empty()) return false;
  for (uint8_t b : name) {
    if (b < 32 || b > 126) return false;
  }
  return !NameBytesEmitAsUndoUnquotedLiteral(name);
}

void AppendFrameNameTokenForUndo(std::string* cmd,
                                 const std::vector<uint8_t>& name) {
  if (name.empty()) return;
  if (NameBytesEmitAsUndoUnquotedLiteral(name)) {
    for (uint8_t b : name) {
      cmd->push_back(static_cast<char>(b));
    }
  } else if (NameBytesEmitAsUndoQuotedAscii(name)) {
    cmd->push_back('"');
    for (uint8_t b : name) {
      if (b == '\\' || b == '"') {
        cmd->push_back('\\');
      }
      cmd->push_back(static_cast<char>(b));
    }
    cmd->push_back('"');
  } else {
    *cmd += HexLowerFromUtf8(name);
  }
}

static const char* BlendModeSlug(uint32_t m) {
  switch (m) {
    case 0:
      return "replace";
    case 1:
      return "add";
    case 2:
      return "blend";
    case 3:
      return "alpha_weighted_add";
    case 4:
      return "mul";
    default:
      return "replace";
  }
}

bool FrameDisplayAabb(const ParsedCodestream& cs, size_t frame_index,
                      uint32_t orient, int64_t* out_dx, int64_t* out_dy,
                      uint64_t* out_dw, uint64_t* out_dh) {
  if (frame_index >= cs.frames.size()) return false;
  const FrameHeader& fh = cs.frames[frame_index].frame;
  const uint32_t canvas_w = cs.image.size.width;
  const uint32_t canvas_h = cs.image.size.height;
  int64_t sx_old = 0;
  int64_t sy_old = 0;
  uint64_t rw = canvas_w;
  uint64_t rh = canvas_h;
  if (fh.have_crop) {
    sx_old = UnpackSigned(fh.ux0);
    sy_old = UnpackSigned(fh.uy0);
    rw = fh.crop_width;
    rh = fh.crop_height;
  }
  StorageRectToDisplayAabb(orient, canvas_w, canvas_h, sx_old, sy_old, rw, rh,
                           out_dx, out_dy, out_dw, out_dh);
  return true;
}

bool LogicalSection0BodyByteOffsetRu(const FramedUnit& fu, size_t* out_bytes) {
  if (fu.toc_decoded_sizes.empty()) return false;
  if (fu.toc_perm.empty()) {
    *out_bytes = 0;
    return true;
  }
  for (size_t p = 0; p < fu.toc_perm.size(); ++p) {
    if (fu.toc_perm[p] == 0) {
      size_t off = 0;
      for (size_t q = 0; q < p; ++q) {
        if (q >= fu.toc_decoded_sizes.size()) return false;
        off += fu.toc_decoded_sizes[q];
      }
      *out_bytes = off;
      return true;
    }
  }
  return false;
}

std::string HexLowerPhoton10(const std::array<uint8_t, 10>& b) {
  std::vector<uint8_t> v(b.begin(), b.end());
  return HexLowerFromUtf8(v);
}

// Replace bits [del_start_bit, del_end_bit_excl) in |buf| with
// ins_src[ins_start_bit .. +ins_num_bits). Used so --check_reversible can restore
// the exact pre-transform frame-header encoding (e.g. packed all-default) after
// EPF edits expanded the loop-filter bundle in the working codestream.
bool SpliceCodestreamReplaceBitSpan(std::vector<uint8_t>* buf,
                                    size_t del_start_bit,
                                    size_t del_end_bit_excl,
                                    const uint8_t* ins_src, size_t ins_src_size_bytes,
                                    size_t ins_start_bit, size_t ins_num_bits) {
  if (del_end_bit_excl < del_start_bit) return false;
  const size_t old_bytes = buf->size();
  if (old_bytes > SIZE_MAX / 8u) return false;
  const size_t old_total_bits = old_bytes * 8;
  if (del_end_bit_excl > old_total_bits) return false;
  if (ins_num_bits != 0) {
    if (ins_src_size_bytes > SIZE_MAX / 8u) return false;
    const size_t ins_total_bits = ins_src_size_bytes * 8;
    if (ins_start_bit > ins_total_bits || ins_num_bits > ins_total_bits - ins_start_bit) {
      return false;
    }
  }
  BitWriter bw;
  bw.AppendBitsRange(buf->data(), 0, del_start_bit);
  bw.AppendBitsRange(ins_src, ins_start_bit, ins_num_bits);
  bw.AppendBitsRange(buf->data(), del_end_bit_excl,
                     old_total_bits - del_end_bit_excl);
  bw.ZeroPadToByte();
  *buf = bw.TakeBytes();
  return true;
}

// Frames whose frame headers may need a verbatim splice from the expected
// codestream after --gab-* (same rf_reencode / packed-header issue as EPF).
void CollectGabHeaderSpliceFrameIndices(const ParsedCodestream& exp_cs,
                                       const std::vector<size_t>& selective_frames,
                                       std::vector<size_t>* out) {
  out->clear();
  const auto push_if_editable = [&](size_t fi) {
    if (fi >= exp_cs.frames.size()) return;
    const uint32_t ft = exp_cs.frames[fi].frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) return;
    out->push_back(fi);
  };
  if (!selective_frames.empty()) {
    for (size_t fi : selective_frames) push_if_editable(fi);
  } else {
    for (size_t fi = 0; fi < exp_cs.frames.size(); ++fi) push_if_editable(fi);
  }
  std::sort(out->begin(), out->end());
  out->erase(std::unique(out->begin(), out->end()), out->end());
}

}  // namespace

void UndoRecorder::Reset() {
  *this = UndoRecorder();
}

void UndoRecorder::SetBoxPipelineNotReversible(const char* reason) {
  box_pipeline_ok_ = false;
  if (!block_reason_) block_reason_ = reason;
}

void UndoRecorder::SetCodestreamNotReversible(const char* reason) {
  codestream_ok_ = false;
  if (!block_reason_) block_reason_ = reason;
}

void UndoRecorder::BlockCs(const char* reason) {
  SetCodestreamNotReversible(reason);
}

void UndoRecorder::CapturePreHeader(const ParsedCodestream& parsed) {
  const ImageMetadata& m = parsed.image.metadata;
  orient_before_ = (m.orientation >= 1 && m.orientation <= 8) ? m.orientation : 1;
  if (m.xyb_encoded) {
    have_bits_before_ = true;
    bits_per_sample_before_ = parsed.image.metadata.bit_depth.bits_per_sample;
  } else {
    have_bits_before_ = false;
  }
  if (m.have_animation) {
    have_animation_before_ = true;
    num_loops_before_ = m.animation.num_loops;
    tps_num_before_ = m.animation.tps_numerator;
    tps_den_before_ = m.animation.tps_denominator;
  } else {
    have_animation_before_ = false;
  }
}

void UndoRecorder::NoteCropApplied(uint32_t old_canvas_w, uint32_t old_canvas_h,
                                   int32_t storage_dx, int32_t storage_dy,
                                   uint32_t new_canvas_w, uint32_t new_canvas_h) {
  crop_undo_ = true;
  crop_old_w_ = old_canvas_w;
  crop_old_h_ = old_canvas_h;
  crop_storage_dx_ = storage_dx;
  crop_storage_dy_ = storage_dy;
  crop_new_w_ = new_canvas_w;
  crop_new_h_ = new_canvas_h;
  has_any_step_ = true;
}

void UndoRecorder::NoteGabBlur(float amount) {
  gab_undo_ = true;
  gab_undo_kind_ = GabUndoKind::kSharpen;
  gab_undo_amount_ = amount;
  has_any_step_ = true;
}

void UndoRecorder::NoteGabSharpen(float amount) {
  gab_undo_ = true;
  gab_undo_kind_ = GabUndoKind::kBlur;
  gab_undo_amount_ = amount;
  has_any_step_ = true;
}

void UndoRecorder::NoteEpfAmplitudeScale(float factor) {
  if (factor == 1.f || !std::isfinite(factor)) return;
  epf_amp_undo_ = true;
  epf_amp_forward_factor_ = factor;
  has_any_step_ = true;
}

void UndoRecorder::CaptureRestorationBeforeEpfOps(
    const ParsedCodestream& cs, const std::vector<size_t>* only_frames) {
  epf_rf_before_valid_.assign(cs.frames.size(), false);
  epf_rf_snap_modular_.assign(cs.frames.size(), false);
  epf_rf_before_.resize(cs.frames.size());
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    const FramedUnit& fu = cs.frames[fi];
    const uint32_t ft = fu.frame.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    epf_rf_before_valid_[fi] = true;
    epf_rf_snap_modular_[fi] = (fu.frame.encoding == kFrameEncModular);
    epf_rf_before_[fi] = fu.frame.restoration;
  }
}

void UndoRecorder::FinalizeEpfRestorationUndo(uint32_t mask) {
  epf_undo_mask_ = mask;
  epf_restore_verify_ = (mask != 0u);
  if (!epf_restore_verify_) {
    epf_rf_before_valid_.clear();
    epf_rf_snap_modular_.clear();
    epf_rf_before_.clear();
  }
  if (epf_restore_verify_) {
    has_any_step_ = true;
  }
}

void UndoRecorder::NoteOpsinAdjust(bool exp, bool temp, bool tint, bool hue,
                                   float ev, float temperature, float tint_v,
                                   float hue_v) {
  opsin_undo_ = true;
  opsin_set_exp_ = exp;
  opsin_set_temp_ = temp;
  opsin_set_tint_ = tint;
  opsin_set_hue_ = hue;
  opsin_ev_ = ev;
  opsin_temp_ = temperature;
  opsin_tint_ = tint_v;
  opsin_hue_ = hue_v;
  has_any_step_ = true;
}

bool UndoRecorder::CaptureFrameRegionBeforeMove(
    const ParsedCodestream& parsed, size_t frame_index,
    uint32_t orient_before_header) {
  int64_t dx = 0;
  int64_t dy = 0;
  uint64_t dw = 0;
  uint64_t dh = 0;
  if (!FrameDisplayAabb(parsed, frame_index, orient_before_header, &dx, &dy,
                         &dw, &dh)) {
    return false;
  }
  FrameRegionUndo u;
  u.frame_index = frame_index;
  u.orient_before_header = orient_before_header;
  u.old_display_x = dx;
  u.old_display_y = dy;
  u.display_w = dw;
  u.display_h = dh;
  frame_region_undos_.push_back(std::move(u));
  has_any_step_ = true;
  return true;
}

void UndoRecorder::NoteFrameNameBeforeChange(const ParsedCodestream& parsed,
                                             size_t frame_index) {
  if (frame_index >= parsed.frames.size()) return;
  const FrameHeader& fh = parsed.frames[frame_index].frame;
  FrameNameUndo u;
  u.frame_index = frame_index;
  u.old_name_utf8 = fh.name;
  frame_name_undos_.push_back(std::move(u));
  has_any_step_ = true;
}

void UndoRecorder::NoteSplinesAddedOnFrames(
    const std::vector<std::pair<size_t, int64_t>>& entries) {
  if (entries.empty()) return;
  for (const auto& pr : entries) {
    SplineAddUndoEntry e;
    e.frame_index = pr.first;
    e.forward_body_delta_bits = pr.second;
    e.lf_chunk0_tail_trim_bytes = 0;
    spline_add_undos_.push_back(e);
  }
  has_any_step_ = true;
}

void UndoRecorder::FinalizeSplineLfGlobalChunk0TailTrim(
    const uint8_t* ref_cs, size_t ref_n, const uint8_t* out_cs, size_t out_n) {
  if (spline_add_undos_.empty()) return;
  ParsedCodestream ref_parsed;
  ParsedCodestream out_parsed;
  if (!ReadCodestream(ref_cs, ref_n, &ref_parsed) ||
      !ReadCodestream(out_cs, out_n, &out_parsed)) {
    return;
  }
  for (SplineAddUndoEntry& e : spline_add_undos_) {
    e.lf_chunk0_tail_trim_bytes = ComputeLfGlobalChunk0TailPaddingTrimBytes(
        out_cs, out_n, out_parsed, ref_cs, ref_n, ref_parsed, e.frame_index);
  }
}

void UndoRecorder::CapturePhotonNoiseBeforeApply(
    const ParsedCodestream& cs, const std::vector<size_t>* only_frames) {
  photon_before_by_frame_.clear();
  photon_before_by_frame_.resize(cs.frames.size());
  const auto want_frame = [&](size_t idx) -> bool {
    if (only_frames == nullptr || only_frames->empty()) return true;
    return std::binary_search(only_frames->begin(), only_frames->end(), idx);
  };
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    if (!want_frame(fi)) continue;
    const FramedUnit& fu = cs.frames[fi];
    const FrameHeader& fh = fu.frame;
    const uint32_t ft = fh.frame_type;
    if (ft == kFrameTypeLF || ft == kFrameTypeReferenceOnly) continue;
    if (ft != kFrameTypeRegular && ft != kFrameTypeSkipProgressive) continue;
    size_t sec0 = 0;
    if (!LogicalSection0BodyByteOffsetRu(fu, &sec0)) continue;
    if (fu.body_bit_length == 0) continue;
    PhotonBeforeSnap& s = photon_before_by_frame_[fi];
    s.had_noise = (fh.flags & kFrameFlagNoise) != 0;
    if (s.had_noise && fu.lf_global_noise_raw_valid) {
      s.raw_valid = true;
      s.raw_bytes = fu.lf_global_noise_raw_bytes;
    }
  }
}

void UndoRecorder::FinalizePhotonNoiseAfterApply(const ParsedCodestream& cs) {
  photon_undos_.clear();
  for (size_t fi = 0; fi < cs.frames.size(); ++fi) {
    const FramedUnit& fu = cs.frames[fi];
    if (!fu.photon_noise_edit) continue;
    PhotonNoiseUndoSpec u;
    u.frame_index = fi;
    if (fi >= photon_before_by_frame_.size()) {
      SetCodestreamNotReversible("photon noise: internal undo snapshot mismatch");
      return;
    }
    const PhotonBeforeSnap& snap = photon_before_by_frame_[fi];
    if (fu.photon_noise_delta_bytes == 10) {
      u.forward_kind = PhotonNoiseForwardKind::kInsert;
    } else if (fu.photon_noise_delta_bytes == -10) {
      u.forward_kind = PhotonNoiseForwardKind::kRemove;
      if (!snap.had_noise || !snap.raw_valid) {
        SetCodestreamNotReversible(
            "photon noise: internal undo state for removed LUT");
        return;
      }
      u.prior_bytes = snap.raw_bytes;
    } else if (fu.photon_noise_delta_bytes == 0) {
      u.forward_kind = PhotonNoiseForwardKind::kReplace;
      if (!snap.had_noise || !snap.raw_valid) {
        SetCodestreamNotReversible(
            "photon noise: internal undo state for replaced LUT");
        return;
      }
      u.prior_bytes = snap.raw_bytes;
    } else {
      SetCodestreamNotReversible("photon noise: unknown splice delta for undo");
      return;
    }
    photon_undos_.push_back(u);
  }
  photon_before_by_frame_.clear();
  if (!photon_undos_.empty()) {
    has_any_step_ = true;
  }
}

void UndoRecorder::NoteFrameBlendBeforeOverrides(
    const ParsedCodestream& parsed,
    const std::vector<FrameBlendOverride>& overrides) {
  if (overrides.empty()) return;
  std::vector<bool> seen(parsed.frames.size(), false);
  const uint32_t num_extra = parsed.image.metadata.num_extra;
  for (const FrameBlendOverride& o : overrides) {
    if (o.frame_index >= parsed.frames.size()) continue;
    if (seen[o.frame_index]) continue;
    seen[o.frame_index] = true;
    const FrameHeader& fh = parsed.frames[o.frame_index].frame;
    BlendUndo u;
    u.frame_index = o.frame_index;
    u.num_extra = num_extra;
    u.is_last = fh.is_last;
    u.main = fh.blending_info;
    u.ec = fh.ec_blending_info;
    u.save_as_reference = fh.save_as_reference;
    blend_undos_.push_back(std::move(u));
    has_any_step_ = true;
  }
}

void UndoRecorder::NoteKeepReorder(KeepReorderUndoSpec spec) {
  if (spec.forward_order.empty()) return;
  keep_reorder_undo_ = true;
  keep_reorder_spec_ = std::move(spec);
  has_any_step_ = true;
}

void UndoRecorder::NoteHeaderChanges(bool set_abs_orient_nonzero,
                                     bool set_rel_nonzero, uint32_t orient_after,
                                     bool set_bits_nonzero, bool set_loops,
                                     bool set_tps) {
  if (set_abs_orient_nonzero || set_rel_nonzero) {
    if (orient_after != orient_before_) {
      undo_orientation_ = true;
      has_any_step_ = true;
    }
  }
  if (set_bits_nonzero) {
    undo_bits_ = true;
    has_any_step_ = true;
  }
  if (set_loops) {
    undo_loops_ = true;
    has_any_step_ = true;
  }
  if (set_tps) {
    undo_tps_ = true;
    has_any_step_ = true;
  }
}

void UndoRecorder::SetSelectiveFramesCopy(const std::vector<size_t>& indices) {
  selective_frames_ = indices;
}

void UndoRecorder::FinalizePipelineCompatibility() {
  if (crop_undo_ && !frame_region_undos_.empty()) {
    SetCodestreamNotReversible(
        "--crop combined with --set-frame-region (multi-pass undo not emitted)");
  }
  if (crop_undo_ && (gab_undo_ || epf_amp_undo_ || epf_restore_verify_)) {
    SetCodestreamNotReversible(
        "--crop combined with --gab-* or --set-epf-*");
  }
  if (crop_undo_ && !spline_add_undos_.empty()) {
    SetCodestreamNotReversible(
        "--crop combined with --set-splines-from (add-only undo not emitted)");
  }
  if (!frame_region_undos_.empty() && !spline_add_undos_.empty()) {
    SetCodestreamNotReversible(
        "--set-frame-region combined with --set-splines-from (add-only undo not "
        "emitted)");
  }
  for (const BlendUndo& b : blend_undos_) {
    for (size_t e = 0; e < b.ec.size(); ++e) {
      const FrameBlendingInfo& bi = b.ec[e];
      if (bi.mode != b.main.mode || bi.alpha_channel != b.main.alpha_channel ||
          bi.clamp != b.main.clamp || bi.source != b.main.source) {
        SetCodestreamNotReversible(
            "--set-frame-blends: extra-channel blending differs from main "
            "(undo not emitted)");
        return;
      }
    }
  }
}

void UndoRecorder::CaptureTocBeforeGroupOrder(
    const ParsedCodestream& cs, const std::vector<size_t>* only_frames) {
  if (!CaptureTocSnapshotsBeforeGroupOrder(cs, only_frames, &toc_snaps_)) {
    toc_snaps_.clear();
  }
}

void UndoRecorder::FinalizeTocGroupOrderAfterApply(bool toc_changed) {
  if (!toc_changed) {
    toc_snaps_.clear();
    return;
  }
  toc_undo_ = true;
  has_any_step_ = true;
}

bool UndoRecorder::BuildUndoCommandLine(const char* argv0, const char* out_jxl,
                                        const char* restored_jxl,
                                        std::string* out_line) const {
  out_line->clear();
  if (!box_pipeline_ok_ || !codestream_ok_ || !has_any_step_) return false;
  if (toc_undo_) {
    // TOC layout is restored from an internal snapshot for --check_reversible;
    // there is no single argv inverse that round-trips arbitrary permutations.
    return false;
  }
  std::string cmd;
  cmd += ShellQuotePath(argv0);
  cmd += ' ';
  cmd += ShellQuotePath(out_jxl);

  if (keep_reorder_undo_) {
    cmd += " --keep-listed-frames --keep-frames=";
    const std::vector<size_t>& fw = keep_reorder_spec_.forward_order;
    const size_t n = fw.size();
    std::vector<size_t> inv(n);
    for (size_t j = 0; j < n; ++j) {
      inv[fw[j]] = j;
    }
    for (size_t i = 0; i < n; ++i) {
      if (i != 0) cmd += ',';
      char b[32];
      snprintf(b, sizeof(b), "%" PRIuS, inv[i]);
      cmd += b;
    }
  }

  if (undo_orientation_) {
    char buf[64];
    snprintf(buf, sizeof(buf), " --set-orientation=%" PRIu32, orient_before_);
    cmd += buf;
  }
  if (undo_bits_ && have_bits_before_) {
    char buf[64];
    snprintf(buf, sizeof(buf), " --set-bits-per-sample=%" PRIu32,
             bits_per_sample_before_);
    cmd += buf;
  }
  if (undo_loops_ && have_animation_before_) {
    char buf[64];
    snprintf(buf, sizeof(buf), " --set-num-loops=%" PRIu32, num_loops_before_);
    cmd += buf;
  }
  if (undo_tps_ && have_animation_before_) {
    char buf[80];
    snprintf(buf, sizeof(buf), " --set-tps=%" PRIu32 "/%" PRIu32, tps_num_before_,
             tps_den_before_);
    cmd += buf;
  }

  if (opsin_undo_) {
    cmd += " --opsin-inverse";
    char buf[128];
    if (opsin_set_exp_) {
      snprintf(buf, sizeof(buf), " --opsin-exposure=%.9g", opsin_ev_);
      cmd += buf;
    }
    if (opsin_set_temp_) {
      snprintf(buf, sizeof(buf), " --opsin-temperature=%.9g", opsin_temp_);
      cmd += buf;
    }
    if (opsin_set_tint_) {
      snprintf(buf, sizeof(buf), " --opsin-tint=%.9g", opsin_tint_);
      cmd += buf;
    }
    if (opsin_set_hue_) {
      snprintf(buf, sizeof(buf), " --opsin-hue=%.9g", opsin_hue_);
      cmd += buf;
    }
  }

  for (size_t fri = 0; fri < frame_region_undos_.size(); ++fri) {
    const FrameRegionUndo& fr = frame_region_undos_[fri];
    cmd += " --set-frame-region=";
    char buf[160];
    snprintf(buf, sizeof(buf),
             "%" PRIuS ":%" PRIu64 "x%" PRIu64 "%+" PRId64 "%+" PRId64,
             fr.frame_index, fr.display_w, fr.display_h, fr.old_display_x,
             fr.old_display_y);
    cmd += buf;
  }

  if (crop_undo_) {
    char buf[128];
    snprintf(buf, sizeof(buf), " --crop=%" PRIu32 "x%" PRIu32 "%+" PRId32 "%+" PRId32,
             crop_old_w_, crop_old_h_, -crop_storage_dx_, -crop_storage_dy_);
    cmd += buf;
  }

  if (gab_undo_) {
    char buf[64];
    if (gab_undo_kind_ == GabUndoKind::kSharpen) {
      snprintf(buf, sizeof(buf), " --gab-sharpen=%.9g", gab_undo_amount_);
    } else {
      snprintf(buf, sizeof(buf), " --gab-blur=%.9g", gab_undo_amount_);
    }
    cmd += buf;
  }

  if (epf_restore_verify_) {
    if (epf_undo_mask_ == 2u && epf_amp_undo_) {
      char buf[80];
      const float inv = 1.0f / epf_amp_forward_factor_;
      snprintf(buf, sizeof(buf), " --set-epf-amplitude-scale=%.9g", inv);
      cmd += buf;
    } else if (epf_undo_mask_ == 1u) {
      uint32_t prior_iters = 0;
      bool seen_any = false;
      for (size_t fi = 0; fi < epf_rf_before_valid_.size(); ++fi) {
        if (!epf_rf_before_valid_[fi]) continue;
        const ParsedRestorationFilter& s = epf_rf_before_[fi];
        const bool modular = epf_rf_snap_modular_[fi];
        const uint32_t it = std::min(
            3u, s.all_default ? DefaultRestorationFilter(modular).epf_iters
                              : s.epf_iters);
        if (!seen_any) {
          prior_iters = it;
          seen_any = true;
        } else if (prior_iters != it) {
          return false;
        }
      }
      if (!seen_any) return false;
      char buf[64];
      snprintf(buf, sizeof(buf), " --set-epf-iters=%" PRIu32, prior_iters);
      cmd += buf;
      if (!selective_frames_.empty()) {
        cmd += " --frames=";
        for (size_t i = 0; i < selective_frames_.size(); ++i) {
          if (i != 0) cmd += ',';
          char fbuf[32];
          snprintf(fbuf, sizeof(fbuf), "%" PRIuS, selective_frames_[i]);
          cmd += fbuf;
        }
      }
    } else {
      return false;
    }
  }

  if (!frame_name_undos_.empty()) {
    cmd += " --set-frame-names=";
    for (ssize_t i = static_cast<ssize_t>(frame_name_undos_.size()) - 1; i >= 0;
         --i) {
      const FrameNameUndo& fn = frame_name_undos_[static_cast<size_t>(i)];
      if (i != static_cast<ssize_t>(frame_name_undos_.size()) - 1) {
        cmd += ',';
      }
      char idxbuf[32];
      snprintf(idxbuf, sizeof(idxbuf), "%" PRIuS ":", fn.frame_index);
      cmd += idxbuf;
      AppendFrameNameTokenForUndo(&cmd, fn.old_name_utf8);
    }
  }

  if (!blend_undos_.empty()) {
    std::vector<size_t> ord(blend_undos_.size());
    for (size_t i = 0; i < ord.size(); ++i) ord[i] = i;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
      return blend_undos_[a].frame_index < blend_undos_[b].frame_index;
    });
    cmd += " --set-frame-blends=";
    for (size_t k = 0; k < ord.size(); ++k) {
      const BlendUndo& b = blend_undos_[ord[k]];
      if (k != 0) cmd += ',';
      char idxbuf[32];
      snprintf(idxbuf, sizeof(idxbuf), "%" PRIuS ":", b.frame_index);
      cmd += idxbuf;
      cmd += BlendModeSlug(b.main.mode);
      if (b.num_extra > 0) {
        char fbuf[160];
        snprintf(fbuf, sizeof(fbuf), ",alpha=%" PRIu32 ",clamp=%s,source=%" PRIu32,
                 b.main.alpha_channel, b.main.clamp ? "true" : "false",
                 b.main.source);
        cmd += fbuf;
      }
      if (!b.is_last) {
        char tbuf[48];
        snprintf(tbuf, sizeof(tbuf), ",target=%" PRIu32,
                 b.save_as_reference & 3u);
        cmd += tbuf;
      }
    }
  }

  if (!spline_add_undos_.empty()) {
    std::vector<size_t> u;
    u.reserve(spline_add_undos_.size());
    for (const SplineAddUndoEntry& e : spline_add_undos_) {
      u.push_back(e.frame_index);
    }
    std::sort(u.begin(), u.end());
    u.erase(std::unique(u.begin(), u.end()), u.end());
    cmd += " --clear-splines-frames=";
    for (size_t i = 0; i < u.size(); ++i) {
      if (i != 0) cmd += ',';
      char buf[32];
      snprintf(buf, sizeof(buf), "%" PRIuS, u[i]);
      cmd += buf;
    }
    bool any_trim = false;
    for (const SplineAddUndoEntry& e : spline_add_undos_) {
      if (e.lf_chunk0_tail_trim_bytes > 0) {
        any_trim = true;
        break;
      }
    }
    if (any_trim) {
      cmd += " --lf-global-chunk0-tail-trim-bytes=";
      bool first_tp = true;
      for (const SplineAddUndoEntry& e : spline_add_undos_) {
        if (e.lf_chunk0_tail_trim_bytes == 0) continue;
        if (!first_tp) {
          cmd += ',';
        }
        first_tp = false;
        char tbuf[48];
        snprintf(tbuf, sizeof(tbuf), "%" PRIuS ":%u", e.frame_index,
                 static_cast<unsigned>(e.lf_chunk0_tail_trim_bytes));
        cmd += tbuf;
      }
    }
  }

  if (!photon_undos_.empty()) {
    std::vector<size_t> fidx;
    fidx.reserve(photon_undos_.size());
    for (const PhotonNoiseUndoSpec& u : photon_undos_) {
      fidx.push_back(u.frame_index);
    }
    std::sort(fidx.begin(), fidx.end());
    fidx.erase(std::unique(fidx.begin(), fidx.end()), fidx.end());
    auto append_photon_frames = [&]() {
      cmd += " --frames=";
      for (size_t i = 0; i < fidx.size(); ++i) {
        if (i != 0) cmd += ',';
        char b[32];
        snprintf(b, sizeof(b), "%" PRIuS, fidx[i]);
        cmd += b;
      }
    };
    const bool all_insert_forward =
        std::all_of(photon_undos_.begin(), photon_undos_.end(),
                    [](const PhotonNoiseUndoSpec& u) {
                      return u.forward_kind == PhotonNoiseForwardKind::kInsert;
                    });
    if (all_insert_forward) {
      cmd += " --set-photon-noise-iso=0";
      append_photon_frames();
    } else {
      const PhotonNoiseForwardKind k0 = photon_undos_[0].forward_kind;
      if (k0 != PhotonNoiseForwardKind::kRemove &&
          k0 != PhotonNoiseForwardKind::kReplace) {
        return false;
      }
      for (const PhotonNoiseUndoSpec& u : photon_undos_) {
        if (u.forward_kind != k0) return false;
      }
      const std::array<uint8_t, 10>& ref = photon_undos_[0].prior_bytes;
      for (const PhotonNoiseUndoSpec& u : photon_undos_) {
        if (u.prior_bytes != ref) return false;
      }
      cmd += " --set-photon-noise-weights=";
      cmd += HexLowerPhoton10(ref);
      append_photon_frames();
    }
  }

  cmd += ' ';
  cmd += ShellQuotePath(restored_jxl);
  *out_line = std::move(cmd);
  return true;
}

ReversibleCheckStatus UndoRecorder::VerifyRoundtripCodestream(
    const std::vector<uint8_t>& expected_codestream,
    const std::vector<uint8_t>& out_codestream) const {
  if (!codestream_ok_ || !box_pipeline_ok_) {
    return ReversibleCheckStatus::kNotReversibleSkipped;
  }
  if (!has_any_step_) {
    if (expected_codestream.size() != out_codestream.size()) {
      return ReversibleCheckStatus::kMismatch;
    }
    if (memcmp(expected_codestream.data(), out_codestream.data(),
               expected_codestream.size()) != 0) {
      return ReversibleCheckStatus::kMismatch;
    }
    return ReversibleCheckStatus::kOk;
  }
  jxltran::ParsedCodestream parsed;
  if (!jxltran::ReadCodestream(out_codestream.data(), out_codestream.size(),
                               &parsed)) {
    return ReversibleCheckStatus::kMismatch;
  }

  if (keep_reorder_undo_) {
    if (!ApplyKeepListedFramesReorderInverse(&parsed, keep_reorder_spec_)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  if (undo_orientation_ || undo_bits_ || undo_loops_ || undo_tps_) {
    HeaderMod mod;
    mod.set_orientation = 0;
    if (undo_orientation_) {
      mod.set_orientation = orient_before_;
    }
    mod.set_bits_per_sample =
        undo_bits_ && have_bits_before_ ? bits_per_sample_before_ : 0;
    mod.have_set_num_loops = undo_loops_ && have_animation_before_;
    mod.set_num_loops = num_loops_before_;
    mod.have_set_tps = undo_tps_ && have_animation_before_;
    mod.set_tps_numerator = tps_num_before_;
    mod.set_tps_denominator = tps_den_before_;
    if (!ApplyHeaderMod(&parsed, mod)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  if (opsin_undo_) {
    OpsinAdjustParams oa;
    oa.undo_inverse = true;
    if (opsin_set_exp_) oa.exposure_ev = opsin_ev_;
    if (opsin_set_temp_) oa.temperature = opsin_temp_;
    if (opsin_set_tint_) oa.tint = opsin_tint_;
    if (opsin_set_hue_) oa.hue = opsin_hue_;
    bool changed = false;
    if (!ApplyOpsinAdjust(&parsed, oa, &changed)) {
      return ReversibleCheckStatus::kMismatch;
    }
    (void)changed;
  }

  for (ssize_t fri = static_cast<ssize_t>(frame_region_undos_.size()) - 1;
       fri >= 0; --fri) {
    const FrameRegionUndo& fr = frame_region_undos_[static_cast<size_t>(fri)];
    bool ch = false;
    if (!ApplySetFrameRegionFromDisplay(
            &parsed, fr.frame_index, fr.orient_before_header,
            static_cast<int32_t>(fr.old_display_x),
            static_cast<int32_t>(fr.old_display_y),
            static_cast<uint32_t>(fr.display_w), static_cast<uint32_t>(fr.display_h),
            &ch)) {
      return ReversibleCheckStatus::kMismatch;
    }
    (void)ch;
  }

  if (crop_undo_) {
    if (!ApplyCrop(&parsed, -crop_storage_dx_, -crop_storage_dy_, crop_old_w_,
                   crop_old_h_)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  if (toc_undo_) {
    for (const FramedUnitTocSnapshot& snap : toc_snaps_) {
      if (snap.frame_index >= parsed.frames.size()) {
        return ReversibleCheckStatus::kMismatch;
      }
      RestoreFramedUnitTocSnapshot(snap, &parsed.frames[snap.frame_index]);
    }
  }

  if (!photon_undos_.empty()) {
    for (ssize_t i = static_cast<ssize_t>(photon_undos_.size()) - 1; i >= 0; --i) {
      const PhotonNoiseUndoSpec& u = photon_undos_[static_cast<size_t>(i)];
      if (!ApplyPhotonNoiseUndo(&parsed, u)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  if (epf_restore_verify_) {
    for (size_t fi = 0; fi < epf_rf_before_valid_.size(); ++fi) {
      if (!epf_rf_before_valid_[fi]) continue;
      if (!ApplyRestorationFilterExactRestore(&parsed, fi, epf_rf_before_[fi])) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  const std::vector<size_t>* sel =
      selective_frames_.empty() ? nullptr : &selective_frames_;

  if (gab_undo_) {
    GabArgs ga;
    if (gab_undo_kind_ == GabUndoKind::kSharpen) {
      ga.kind = GabArgs::Kind::kSharpen;
    } else {
      ga.kind = GabArgs::Kind::kBlur;
    }
    ga.amount = gab_undo_amount_;
    if (!ApplyGabArgs(&parsed, ga, sel)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  if (!blend_undos_.empty()) {
    std::vector<size_t> ord(blend_undos_.size());
    for (size_t j = 0; j < ord.size(); ++j) ord[j] = j;
    std::sort(ord.begin(), ord.end(), [&](size_t a, size_t b) {
      return blend_undos_[a].frame_index < blend_undos_[b].frame_index;
    });
    for (size_t j = 0; j < ord.size(); ++j) {
      const BlendUndo& b = blend_undos_[ord[j]];
      if (!ApplyFrameBlendExactRestore(&parsed, b.frame_index, b.main, b.ec,
                                       b.save_as_reference)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  if (!spline_add_undos_.empty()) {
    for (ssize_t i = static_cast<ssize_t>(spline_add_undos_.size()) - 1; i >= 0;
         --i) {
      const SplineAddUndoEntry& e = spline_add_undos_[static_cast<size_t>(i)];
      if (!ApplyRemoveSplinesAddOnly(&parsed, e.frame_index,
                                     &e.forward_body_delta_bits)) {
        return ReversibleCheckStatus::kMismatch;
      }
      if (!ApplyLfGlobalPhysicalChunk0TailTrim(
              &parsed, e.frame_index, e.lf_chunk0_tail_trim_bytes)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  if (!frame_name_undos_.empty()) {
    std::vector<std::pair<size_t, std::vector<uint8_t>>> pairs;
    pairs.reserve(frame_name_undos_.size());
    for (ssize_t i = static_cast<ssize_t>(frame_name_undos_.size()) - 1; i >= 0;
         --i) {
      const FrameNameUndo& fn = frame_name_undos_[static_cast<size_t>(i)];
      pairs.emplace_back(fn.frame_index, fn.old_name_utf8);
    }
    if (!ApplySetFrameNames(&parsed, pairs)) {
      return ReversibleCheckStatus::kMismatch;
    }
  }

  std::vector<uint8_t> restored;
  if (!WriteCodestream(parsed, out_codestream.data(), &restored)) {
    return ReversibleCheckStatus::kMismatch;
  }

  // WriteCodestream re-encodes the image header from structs; alternate valid
  // entropy codings can differ bitwise from the pre-transform codestream. For
  // --check_reversible, keep the original image-header prefix (through the byte
  // before the first frame) when the first-frame layout matches.
  {
    ParsedCodestream exp_h, rest_h;
    if (!ReadCodestream(expected_codestream.data(), expected_codestream.size(),
                        &exp_h) ||
        !ReadCodestream(restored.data(), restored.size(), &rest_h)) {
      return ReversibleCheckStatus::kMismatch;
    }
    if (exp_h.frames.empty() || rest_h.frames.empty()) {
      return ReversibleCheckStatus::kMismatch;
    }
    const size_t hdr_e = exp_h.frames[0].original_frame_byte_offset;
    const size_t hdr_r = rest_h.frames[0].original_frame_byte_offset;
    if (hdr_e != hdr_r) {
      return ReversibleCheckStatus::kMismatch;
    }
    if (hdr_e > restored.size() || hdr_e > expected_codestream.size()) {
      return ReversibleCheckStatus::kMismatch;
    }
    memcpy(restored.data(), expected_codestream.data(), hdr_e);
  }

  if (epf_restore_verify_ || gab_undo_ || toc_undo_ ||
      !spline_add_undos_.empty()) {
    ParsedCodestream exp_cs;
    if (!ReadCodestream(expected_codestream.data(), expected_codestream.size(),
                         &exp_cs)) {
      return ReversibleCheckStatus::kMismatch;
    }
    const auto sort_frame_index_desc = [&](std::vector<size_t>* v) {
      std::sort(v->begin(), v->end(), [&](size_t a, size_t b) {
        if (a >= exp_cs.frames.size() || b >= exp_cs.frames.size()) {
          return a > b;
        }
        return exp_cs.frames[a].original_frame_byte_offset >
               exp_cs.frames[b].original_frame_byte_offset;
      });
    };

    std::vector<size_t> whole_frame;
    if (gab_undo_) {
      CollectGabHeaderSpliceFrameIndices(exp_cs, selective_frames_, &whole_frame);
    }
    if (toc_undo_) {
      for (const FramedUnitTocSnapshot& snap : toc_snaps_) {
        whole_frame.push_back(snap.frame_index);
      }
    }
    if (!spline_add_undos_.empty()) {
      for (const SplineAddUndoEntry& e : spline_add_undos_) {
        whole_frame.push_back(e.frame_index);
      }
    }
    std::sort(whole_frame.begin(), whole_frame.end());
    whole_frame.erase(std::unique(whole_frame.begin(), whole_frame.end()),
                      whole_frame.end());
    const std::vector<size_t> whole_frame_asc = whole_frame;
    sort_frame_index_desc(&whole_frame);

    for (size_t k = 0; k < whole_frame.size(); ++k) {
      const size_t fi = whole_frame[k];
      ParsedCodestream cur_cs;
      if (!ReadCodestream(restored.data(), restored.size(), &cur_cs)) {
        return ReversibleCheckStatus::kMismatch;
      }
      if (fi >= cur_cs.frames.size() || fi >= exp_cs.frames.size()) {
        return ReversibleCheckStatus::kMismatch;
      }
      const FramedUnit& rfu = cur_cs.frames[fi];
      const FramedUnit& efu = exp_cs.frames[fi];
      const size_t r0 = rfu.original_frame_byte_offset * 8;
      const size_t r1 = r0 + rfu.full_frame_byte_len * 8;
      const size_t e0 = efu.original_frame_byte_offset * 8;
      const size_t e1 = e0 + efu.full_frame_byte_len * 8;
      if (r1 < r0 || e1 < e0) return ReversibleCheckStatus::kMismatch;
      if (!SpliceCodestreamReplaceBitSpan(
              &restored, r0, r1, expected_codestream.data(),
              expected_codestream.size(), e0, e1 - e0)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }

    std::vector<size_t> epf_only;
    if (epf_restore_verify_) {
      for (size_t fi = 0; fi < epf_rf_before_valid_.size(); ++fi) {
        if (!epf_rf_before_valid_[fi]) continue;
        if (std::binary_search(whole_frame_asc.begin(), whole_frame_asc.end(),
                               fi)) {
          continue;
        }
        epf_only.push_back(fi);
      }
    }
    sort_frame_index_desc(&epf_only);
    for (size_t k = 0; k < epf_only.size(); ++k) {
      const size_t fi = epf_only[k];
      ParsedCodestream cur_cs;
      if (!ReadCodestream(restored.data(), restored.size(), &cur_cs)) {
        return ReversibleCheckStatus::kMismatch;
      }
      if (fi >= cur_cs.frames.size() || fi >= exp_cs.frames.size()) {
        return ReversibleCheckStatus::kMismatch;
      }
      const FramedUnit& rfu = cur_cs.frames[fi];
      const FramedUnit& efu = exp_cs.frames[fi];
      const size_t r0 = rfu.original_frame_byte_offset * 8;
      const size_t r1 = r0 + rfu.toc_start_bit;
      const size_t e0 = efu.original_frame_byte_offset * 8;
      const size_t e1 = e0 + efu.toc_start_bit;
      if (r1 < r0 || e1 < e0) return ReversibleCheckStatus::kMismatch;
      if (!SpliceCodestreamReplaceBitSpan(
              &restored, r0, r1, expected_codestream.data(),
              expected_codestream.size(), e0, e1 - e0)) {
        return ReversibleCheckStatus::kMismatch;
      }
    }
  }

  if (restored.size() != expected_codestream.size()) {
    return ReversibleCheckStatus::kMismatch;
  }
  if (memcmp(restored.data(), expected_codestream.data(), restored.size()) !=
      0) {
    return ReversibleCheckStatus::kMismatch;
  }
  return ReversibleCheckStatus::kOk;
}

bool ArgsBoxPipelineReversibleForUndo(bool strip_nonzero, bool jxlp_non_keep,
                                      bool box_order_non_keep, bool brob_non_keep,
                                      bool append_any, bool meta_set_any) {
  return !strip_nonzero && !jxlp_non_keep && !box_order_non_keep &&
         !brob_non_keep && !append_any && !meta_set_any;
}

}  // namespace jxltran
