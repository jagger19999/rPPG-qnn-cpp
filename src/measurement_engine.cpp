#include "rppg_qnn/measurement_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace rppg_qnn {
namespace {

double clamp01(double value) { return std::clamp(value, 0.0, 1.0); }

void add_flag(std::vector<std::string>* flags, const std::string& flag) {
  if (std::find(flags->begin(), flags->end(), flag) == flags->end()) {
    flags->push_back(flag);
  }
}

bool finite_candidate(const CandidateResult& candidate) {
  return candidate.valid && std::isfinite(candidate.bpm) && candidate.bpm > 0.0 &&
         std::isfinite(candidate.confidence) &&
         std::isfinite(candidate.peak_ratio);
}

struct Consensus {
  bool valid{false};
  double bpm{0.0};
  double confidence{0.0};
  double spread{0.0};
  std::string selected_method;
  std::vector<std::string> methods;
  std::string artifact_correction;
  std::string rejection_reason;
};

struct CandidateVariant {
  const CandidateResult* candidate{nullptr};
  double bpm{0.0};
  std::string correction;
};

double consensus_confidence(const std::vector<CandidateVariant>& selected,
                            std::size_t total_methods,
                            double maximum_spread) {
  const auto [minimum, maximum] = std::minmax_element(
      selected.begin(), selected.end(),
      [](const CandidateVariant& left, const CandidateVariant& right) {
        return left.bpm < right.bpm;
      });
  const double spread = maximum->bpm - minimum->bpm;
  const double agreement = static_cast<double>(selected.size()) /
                           static_cast<double>(std::max<std::size_t>(1, total_methods));
  const double spread_score = clamp01(1.0 - spread / maximum_spread);
  double spectral = 0.0;
  std::size_t correction_count = 0;
  for (const CandidateVariant& variant : selected) {
    spectral += clamp01(variant.candidate->confidence);
    if (!variant.correction.empty()) ++correction_count;
  }
  spectral /= static_cast<double>(selected.size());
  return clamp01(0.45 * agreement + 0.35 * spread_score + 0.20 * spectral -
                 0.06 * static_cast<double>(correction_count));
}

Consensus consensus(const std::vector<CandidateResult>& candidates,
                    double maximum_spread) {
  std::vector<const CandidateResult*> usable;
  std::set<std::string> methods;
  for (const CandidateResult& candidate : candidates) {
    if ((candidate.method == "POS" || candidate.method == "CHROM" ||
         candidate.method == "LGI") &&
        finite_candidate(candidate) && candidate.bpm >= 45.0 * 0.45 &&
        candidate.bpm <= 150.0 * 2.05) {
      usable.push_back(&candidate);
      methods.insert(candidate.method);
    }
  }
  if (usable.size() < 2U) {
    Consensus result;
    result.rejection_reason = "few_valid_methods";
    return result;
  }

  std::vector<std::vector<CandidateVariant>> variants_by_method;
  for (const std::string& method : methods) {
    std::vector<CandidateVariant> variants;
    const auto found = std::find_if(
        usable.begin(), usable.end(), [&method](const CandidateResult* value) {
          return value->method == method;
        });
    if (found == usable.end()) continue;
    const CandidateResult* value = *found;
    if (value->bpm >= 45.0 && value->bpm <= 150.0) {
      variants.push_back({value, value->bpm, ""});
    }
    if (value->bpm / 2.0 >= 45.0 && value->bpm / 2.0 <= 150.0) {
      variants.push_back({value, value->bpm / 2.0, "halved"});
    }
    if (value->bpm * 2.0 >= 45.0 && value->bpm * 2.0 <= 150.0) {
      variants.push_back({value, value->bpm * 2.0, "doubled"});
    }
    if (!variants.empty()) variants_by_method.push_back(std::move(variants));
  }

  std::vector<CandidateVariant> all_variants;
  for (const auto& variants : variants_by_method) {
    all_variants.insert(all_variants.end(), variants.begin(), variants.end());
  }
  std::vector<CandidateVariant> best;
  double best_score = -1.0;
  const double tolerance = std::min(8.0, maximum_spread);
  for (const CandidateVariant& anchor : all_variants) {
    std::vector<CandidateVariant> selected;
    for (const auto& variants : variants_by_method) {
      const CandidateVariant* nearest = nullptr;
      for (const CandidateVariant& value : variants) {
        if (std::abs(value.bpm - anchor.bpm) > tolerance) continue;
        if (nearest == nullptr ||
            std::abs(value.bpm - anchor.bpm) <
                std::abs(nearest->bpm - anchor.bpm) ||
            (std::abs(value.bpm - anchor.bpm) ==
                 std::abs(nearest->bpm - anchor.bpm) &&
             nearest->correction.size() > value.correction.size())) {
          nearest = &value;
        }
      }
      if (nearest != nullptr) selected.push_back(*nearest);
    }
    if (selected.size() < 2U) continue;
    const auto [minimum, maximum] = std::minmax_element(
        selected.begin(), selected.end(),
        [](const CandidateVariant& left, const CandidateVariant& right) {
          return left.bpm < right.bpm;
        });
    const double spread = maximum->bpm - minimum->bpm;
    const std::size_t correction_count = static_cast<std::size_t>(std::count_if(
        selected.begin(), selected.end(), [](const CandidateVariant& value) {
          return !value.correction.empty();
        }));
    const double score = 2.0 * static_cast<double>(selected.size()) +
                         consensus_confidence(selected, methods.size(), maximum_spread) -
                         0.04 * spread - 0.10 * correction_count;
    if (score > best_score) {
      best_score = score;
      best = std::move(selected);
    }
  }
  if (best.size() < 2U) {
    Consensus result;
    result.rejection_reason = "method_disagreement";
    return result;
  }
  const auto [minimum, maximum] = std::minmax_element(
      best.begin(), best.end(), [](const CandidateVariant& left,
                                  const CandidateVariant& right) {
        return left.bpm < right.bpm;
      });
  const double spread = maximum->bpm - minimum->bpm;
  if (spread > maximum_spread) {
    Consensus result;
    result.rejection_reason = "method_disagreement";
    return result;
  }
  double weighted_bpm = 0.0;
  double weight_sum = 0.0;
  Consensus result;
  result.spread = spread;
  const CandidateVariant* selected_method = &best.front();
  for (const CandidateVariant& value : best) {
    const double weight = std::max(0.01, value.candidate->confidence);
    weighted_bpm += value.bpm * weight;
    weight_sum += weight;
    result.methods.push_back(value.candidate->method);
    if (value.candidate->confidence > selected_method->candidate->confidence) {
      selected_method = &value;
    }
    if (!value.correction.empty()) {
      if (!result.artifact_correction.empty()) result.artifact_correction += ",";
      result.artifact_correction += value.candidate->method + ":" + value.correction;
    }
  }
  result.confidence = consensus_confidence(best, methods.size(), maximum_spread);
  if (result.confidence < 0.20) {
    result.rejection_reason = "low_consensus_confidence";
    return result;
  }
  result.valid = true;
  result.bpm = weighted_bpm / weight_sum;
  result.selected_method = selected_method->candidate->method;
  return result;
}

double quality_score(const QualityMetrics& quality,
                     const QualityProfile& profile,
                     const std::vector<std::string>& flags) {
  double brightness_score = 1.0;
  if (quality.brightness < profile.min_brightness) {
    brightness_score = quality.brightness / std::max(profile.min_brightness, 1e-6);
  } else if (quality.brightness > profile.max_brightness) {
    brightness_score =
        1.0 - (quality.brightness - profile.max_brightness) / 0.08;
  }
  const double signal_score =
      quality.signal_std / std::max(profile.min_signal_std * 5.0, 1e-6);
  const double peak_score = quality.spectral_peak_ratio /
                            std::max(profile.min_peak_ratio * 2.0, 1e-6);
  return clamp01(0.25 * clamp01(brightness_score) +
                 0.25 * clamp01(signal_score) + 0.50 * clamp01(peak_score) -
                 0.16 * static_cast<double>(flags.size()));
}

VqaAssessment make_vqa(const QualityMetrics& quality,
                       const QualityProfile& profile,
                       const Consensus& estimate,
                       const std::vector<std::string>& gate_flags) {
  VqaAssessment vqa;
  if (quality.motion_px > profile.max_motion_px) {
    vqa.head_movement_score = 0.0;
  } else if (quality.motion_px > profile.max_motion_px * 0.55) {
    vqa.head_movement_score = 1.0;
  } else if (quality.motion_px > profile.max_motion_px * 0.20) {
    vqa.head_movement_score = 2.0;
  } else {
    vqa.head_movement_score = 3.0;
  }
  if (quality.brightness < profile.min_brightness ||
      quality.brightness > profile.max_brightness) {
    vqa.illumination_score = 0.0;
  } else if (quality.brightness_std > 0.08) {
    vqa.illumination_score = 1.0;
    add_flag(&vqa.flags, "illumination_unstable");
  } else if (quality.brightness_std > 0.035) {
    vqa.illumination_score = 2.0;
  } else {
    vqa.illumination_score = 3.0;
  }
  if (quality.face_count <= 0 ||
      quality.face_area_ratio < profile.min_face_area_ratio ||
      quality.signal_std < profile.min_signal_std) {
    vqa.skin_score = 0.5;
  } else if (quality.face_count > 1) {
    vqa.skin_score = 1.0;
  } else {
    vqa.skin_score = 2.0;
  }
  if (quality.source_fps < 8.0) {
    vqa.camera_score = 0.0;
  } else if (quality.spectral_peak_ratio < profile.min_peak_ratio) {
    vqa.camera_score = estimate.valid ? 1.0 : 0.25;
  } else {
    vqa.camera_score = estimate.valid ? 2.0 : 1.0;
  }
  vqa.method_consensus_score =
      estimate.valid ? clamp01(1.0 - estimate.spread / profile.max_method_spread_bpm)
                     : 0.0;
  vqa.flags = gate_flags;
  double total = vqa.head_movement_score + vqa.illumination_score +
                 vqa.skin_score + vqa.camera_score +
                 vqa.method_consensus_score;
  if (std::find(vqa.flags.begin(), vqa.flags.end(), "method_disagreement") !=
      vqa.flags.end()) {
    total = std::min(total, 7.0);
  }
  vqa.score_0_10 = std::clamp(total, 0.0, 10.0);
  vqa.label = vqa.score_0_10 >= 8.0   ? "Excellent"
              : vqa.score_0_10 >= 6.0 ? "Good"
              : vqa.score_0_10 >= 4.0 ? "Caution"
                                       : "Poor";
  return vqa;
}

}  // namespace

MeasurementEngine::MeasurementEngine(QualityProfile profile)
    : profile_(std::move(profile)) {}

MeasurementSnapshot MeasurementEngine::evaluate(
    std::vector<CandidateResult> candidates, QualityMetrics quality,
    double window_end_sec) {
  MeasurementSnapshot snapshot;
  snapshot.window_end_sec = window_end_sec;
  snapshot.candidates = std::move(candidates);
  snapshot.quality = quality;

  const Consensus estimate =
      consensus(snapshot.candidates, profile_.max_method_spread_bpm);
  snapshot.consensus_methods = estimate.methods;
  snapshot.consensus_spread_bpm = estimate.spread;
  snapshot.consensus_artifact_correction = estimate.artifact_correction;
  snapshot.consensus_rejection_reason = estimate.rejection_reason;
  if (estimate.valid) {
    snapshot.measured_available = true;
    snapshot.measured_bpm = estimate.bpm;
    snapshot.selected_method = "CONSENSUS";
  }

  std::vector<std::string>& flags = snapshot.gate.flags;
  if (quality.face_count <= 0) add_flag(&flags, "no_face");
  if (quality.face_count > 1) add_flag(&flags, "multiple_faces");
  if (quality.face_area_ratio < profile_.min_face_area_ratio)
    add_flag(&flags, "roi_too_small");
  if (quality.brightness < profile_.min_brightness) add_flag(&flags, "too_dark");
  if (quality.brightness > profile_.max_brightness)
    add_flag(&flags, "too_bright");
  if (quality.signal_std < profile_.min_signal_std)
    add_flag(&flags, "low_signal_std");
  if (quality.spectral_peak_ratio < profile_.min_peak_ratio)
    add_flag(&flags, "weak_spectral_peak");
  if (quality.motion_px > profile_.max_motion_px)
    add_flag(&flags, "high_motion");
  if (quality.source_fps < profile_.min_source_fps) add_flag(&flags, "low_fps");
  if (quality.max_frame_gap_sec > profile_.max_frame_gap_sec)
    add_flag(&flags, "capture_gap");
  if (!estimate.valid) {
    add_flag(&flags, estimate.rejection_reason.empty()
                         ? "method_disagreement"
                         : estimate.rejection_reason);
  }
  if (estimate.valid && last_reliable_bpm_.has_value() &&
      std::abs(estimate.bpm - *last_reliable_bpm_) > profile_.max_bpm_jump) {
    add_flag(&flags, "bpm_jump");
  }
  snapshot.gate.quality_score = quality_score(quality, profile_, flags);
  snapshot.gate.accepted = estimate.valid && flags.empty();
  snapshot.gate.reason = snapshot.gate.accepted ? "accepted" : flags.front();
  snapshot.vqa = make_vqa(quality, profile_, estimate, flags);

  const auto router_started = std::chrono::steady_clock::now();
  const auto recommended = std::max_element(
      snapshot.candidates.begin(), snapshot.candidates.end(),
      [](const CandidateResult& left, const CandidateResult& right) {
        return left.confidence < right.confidence;
      });
  if (recommended != snapshot.candidates.end() && finite_candidate(*recommended)) {
    snapshot.router_shadow.recommended_method = recommended->method;
  }
  snapshot.router_shadow.gate_probability = snapshot.gate.quality_score;
  snapshot.router_shadow.evaluation_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() -
                                                router_started)
          .count();

  if (snapshot.gate.accepted) {
    snapshot.accepted_available = true;
    snapshot.accepted_bpm = estimate.bpm;
    snapshot.display_available = true;
    snapshot.display_bpm = estimate.bpm;
    last_reliable_bpm_ = estimate.bpm;
    last_reliable_timestamp_sec_ = window_end_sec;
  } else if (last_reliable_bpm_.has_value() &&
             last_reliable_timestamp_sec_.has_value() &&
             window_end_sec - *last_reliable_timestamp_sec_ <
                 profile_.hold_last_reliable_sec) {
    snapshot.display_available = true;
    snapshot.display_bpm = *last_reliable_bpm_;
    snapshot.display_is_held = true;
  }
  return snapshot;
}

void MeasurementEngine::reset() {
  last_reliable_bpm_.reset();
  last_reliable_timestamp_sec_.reset();
}

}  // namespace rppg_qnn
