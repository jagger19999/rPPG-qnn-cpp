#include "rppg_qnn/traditional_ensemble.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

namespace rppg_qnn {
namespace {

constexpr double kWindowSeconds = 10.0;

CandidateResult candidate_from(const std::optional<HeartRateResult>& result,
                               const std::string& method) {
  CandidateResult candidate;
  candidate.method = method;
  if (!result.has_value()) {
    candidate.invalid_reason = "sampling";
    return candidate;
  }
  candidate.bpm = result->bpm;
  candidate.confidence = result->confidence;
  candidate.peak_ratio = result->peak_ratio;
  candidate.valid = result->is_valid;
  candidate.invalid_reason = result->invalid_reason;
  candidate.waveform = result->waveform;
  return candidate;
}

}  // namespace

TraditionalEnsemble::TraditionalEnsemble(QualityProfile profile)
    : measurement_(std::move(profile)) {}

void TraditionalEnsemble::add_sample(double timestamp_sec,
                                     const cv::Scalar& mean_bgr,
                                     FrameQualitySample quality) {
  green_.add_sample(timestamp_sec, mean_bgr);
  pos_.add_sample(timestamp_sec, mean_bgr);
  chrom_.add_sample(timestamp_sec, mean_bgr);
  lgi_.add_sample(timestamp_sec, mean_bgr);
  if (!std::isfinite(timestamp_sec) || !std::isfinite(mean_bgr[0]) ||
      !std::isfinite(mean_bgr[1]) || !std::isfinite(mean_bgr[2])) {
    reset();
    return;
  }
  samples_.push_back({timestamp_sec,
                      cv::Vec3d(mean_bgr[2], mean_bgr[1], mean_bgr[0]), quality});
  while (!samples_.empty() &&
         samples_.front().timestamp_sec < timestamp_sec - 30.0) {
    samples_.pop_front();
  }
  if (green_.evaluation_count() != predictor_evaluation_count_) {
    predictor_evaluation_count_ = green_.evaluation_count();
    evaluate(timestamp_sec);
  }
}

void TraditionalEnsemble::evaluate(double timestamp_sec) {
  const double start = timestamp_sec - kWindowSeconds;
  const auto begin = std::lower_bound(
      samples_.begin(), samples_.end(), start,
      [](const Sample& sample, double value) { return sample.timestamp_sec < value; });
  if (begin == samples_.end()) {
    return;
  }
  QualityMetrics quality;
  std::vector<double> brightness_values;
  std::vector<double> green_values;
  double motion_sum = 0.0;
  double area_sum = 0.0;
  double maximum_gap = 0.0;
  int maximum_face_count = 0;
  double previous_timestamp = begin->timestamp_sec;
  std::size_t count = 0;
  for (auto current = begin; current != samples_.end(); ++current) {
    const double brightness =
        (current->rgb[0] + current->rgb[1] + current->rgb[2]) / (3.0 * 255.0);
    brightness_values.push_back(brightness);
    green_values.push_back(current->rgb[1] / 255.0);
    motion_sum += current->quality.motion_px;
    area_sum += current->quality.face_area_ratio;
    maximum_face_count = std::max(maximum_face_count, current->quality.face_count);
    if (count > 0U) {
      maximum_gap = std::max(maximum_gap, current->timestamp_sec - previous_timestamp);
    }
    previous_timestamp = current->timestamp_sec;
    ++count;
  }
  const auto mean_std = [](const std::vector<double>& values) {
    if (values.empty()) return std::pair<double, double>{0.0, 0.0};
    double mean = 0.0;
    for (double value : values) mean += value;
    mean /= static_cast<double>(values.size());
    double variance = 0.0;
    for (double value : values) variance += (value - mean) * (value - mean);
    variance /= static_cast<double>(values.size());
    return std::pair<double, double>{mean, std::sqrt(variance)};
  };
  const auto [brightness, brightness_std] = mean_std(brightness_values);
  const auto [unused_green_mean, signal_std] = mean_std(green_values);
  (void)unused_green_mean;
  quality.brightness = brightness;
  quality.brightness_std = brightness_std;
  quality.signal_std = signal_std;
  quality.face_area_ratio = count > 0U ? area_sum / static_cast<double>(count) : 0.0;
  quality.motion_px = count > 0U ? motion_sum / static_cast<double>(count) : 0.0;
  quality.face_count = maximum_face_count;
  quality.max_frame_gap_sec = maximum_gap;
  const double span = samples_.back().timestamp_sec - begin->timestamp_sec;
  quality.source_fps = span > 0.0 ? static_cast<double>(count - 1U) / span : 0.0;

  std::vector<CandidateResult> candidates{
      candidate_from(green_.latest_result(), "GREEN"),
      candidate_from(pos_.latest_result(), "POS"),
      candidate_from(chrom_.latest_result(), "CHROM"),
      candidate_from(lgi_.latest_result(), "LGI")};
  for (const CandidateResult& candidate : candidates) {
    quality.spectral_peak_ratio =
        std::max(quality.spectral_peak_ratio, candidate.peak_ratio);
  }
  latest_ = measurement_.evaluate(std::move(candidates), quality, timestamp_sec);
  ++evaluation_count_;
}

std::optional<MeasurementSnapshot> TraditionalEnsemble::latest_snapshot() const {
  return latest_;
}

std::size_t TraditionalEnsemble::evaluation_count() const {
  return evaluation_count_;
}

void TraditionalEnsemble::reset() {
  green_.reset();
  pos_.reset();
  chrom_.reset();
  lgi_.reset();
  measurement_.reset();
  samples_.clear();
  latest_.reset();
  evaluation_count_ = 0;
  predictor_evaluation_count_ = 0;
}

}  // namespace rppg_qnn
