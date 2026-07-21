#include "rppg_qnn/green_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace rppg_qnn {
namespace {

constexpr double kHistorySeconds = 30.0;
constexpr double kWindowSeconds = 10.0;
constexpr double kResampleFps = 30.0;
constexpr std::size_t kResampleCount = 300;
constexpr double kMinimumSourceFps = 15.0;
constexpr double kMaximumGapSeconds = 0.75;
constexpr int kMinimumFrequencyBin = 7;
constexpr int kMaximumFrequencyBin = 30;
constexpr double kMinimumConfidence = 0.10;
constexpr double kMinimumBpm = 42.0;
constexpr double kMaximumBpm = 180.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-12;
constexpr double kPeakTieRelativeTolerance = 1e-4;

bool finite_bgr(const cv::Scalar& mean_bgr) {
  return std::isfinite(mean_bgr[0]) && std::isfinite(mean_bgr[1]) &&
         std::isfinite(mean_bgr[2]);
}

HeartRateResult invalid_result(const char* reason) {
  HeartRateResult result;
  result.method = "GREEN";
  result.invalid_reason = reason;
  result.backend = "cpu";
  return result;
}

bool finite_result(const HeartRateResult& result) {
  if (!std::isfinite(result.window_start_sec) ||
      !std::isfinite(result.window_end_sec) || !std::isfinite(result.bpm) ||
      !std::isfinite(result.confidence) || !std::isfinite(result.source_fps) ||
      !std::isfinite(result.max_frame_gap_sec) ||
      !std::isfinite(result.inference_ms)) {
    return false;
  }
  return std::all_of(result.waveform.begin(), result.waveform.end(),
                     [](float value) { return std::isfinite(value); });
}

HeartRateResult finite_failure_result(const HeartRateResult& result,
                                      const char* reason) {
  HeartRateResult safe = invalid_result(reason);
  safe.window_start_sec =
      std::isfinite(result.window_start_sec) ? result.window_start_sec : 0.0;
  safe.window_end_sec =
      std::isfinite(result.window_end_sec) ? result.window_end_sec : 0.0;
  safe.source_fps = std::isfinite(result.source_fps) ? result.source_fps : 0.0;
  safe.source_frame_count = result.source_frame_count;
  safe.max_frame_gap_sec = std::isfinite(result.max_frame_gap_sec)
                                 ? result.max_frame_gap_sec
                                 : 0.0;
  return safe;
}

}  // namespace

void GreenPredictor::add_sample(double timestamp_sec, const cv::Scalar& mean_bgr) {
  if (!std::isfinite(timestamp_sec) || !finite_bgr(mean_bgr) ||
      (!samples_.empty() && timestamp_sec <= samples_.back().timestamp_sec)) {
    reset();
    set_sampling_result();
    return;
  }

  samples_.push_back({timestamp_sec, mean_bgr[1]});
  while (!samples_.empty() &&
         samples_.front().timestamp_sec < timestamp_sec - kHistorySeconds) {
    samples_.pop_front();
  }

  if (samples_.empty() ||
      timestamp_sec - samples_.front().timestamp_sec < kWindowSeconds) {
    set_sampling_result();
    return;
  }

  if (!last_evaluation_timestamp_sec_.has_value() ||
      timestamp_sec - *last_evaluation_timestamp_sec_ >= 1.0) {
    evaluate();
  }
}

std::optional<HeartRateResult> GreenPredictor::latest_result() const {
  return latest_;
}

std::size_t GreenPredictor::buffered_count() const { return samples_.size(); }

double GreenPredictor::buffered_span_sec() const {
  if (samples_.size() < 2U) {
    return 0.0;
  }
  return samples_.back().timestamp_sec - samples_.front().timestamp_sec;
}

std::size_t GreenPredictor::evaluation_count() const { return evaluation_count_; }

void GreenPredictor::reset() {
  samples_.clear();
  latest_.reset();
  last_evaluation_timestamp_sec_.reset();
  evaluation_count_ = 0;
}

void GreenPredictor::set_sampling_result() { latest_ = invalid_result("sampling"); }

void GreenPredictor::evaluate() {
  const double window_end = samples_.back().timestamp_sec;
  const double window_start = window_end - kWindowSeconds;

  HeartRateResult result = invalid_result("sampling");
  result.window_start_sec = window_start;
  result.window_end_sec = window_end;
  const auto lower = std::lower_bound(
      samples_.begin(), samples_.end(), window_start,
      [](const Sample& sample, double timestamp) {
        return sample.timestamp_sec < timestamp;
      });
  if (lower == samples_.end() ||
      (lower == samples_.begin() && lower->timestamp_sec > window_start)) {
    latest_ = result;
    last_evaluation_timestamp_sec_ = window_end;
    ++evaluation_count_;
    return;
  }

  const auto source_begin = lower == samples_.begin() ? lower : std::prev(lower);
  std::vector<Sample> source(source_begin, samples_.end());
  result.source_frame_count = source.size();
  if (source.size() < 2U) {
    latest_ = result;
    last_evaluation_timestamp_sec_ = window_end;
    ++evaluation_count_;
    return;
  }

  const double source_duration = source.back().timestamp_sec - source.front().timestamp_sec;
  if (!std::isfinite(source_duration) || source_duration <= 0.0) {
    latest_ = result;
    last_evaluation_timestamp_sec_ = window_end;
    ++evaluation_count_;
    return;
  }
  result.source_fps =
      static_cast<double>(source.size() - 1U) / source_duration;
  for (std::size_t index = 1; index < source.size(); ++index) {
    result.max_frame_gap_sec = std::max(
        result.max_frame_gap_sec,
        source[index].timestamp_sec - source[index - 1U].timestamp_sec);
  }

  if (result.source_fps < kMinimumSourceFps) {
    result.invalid_reason = "low_source_fps";
  } else if (result.max_frame_gap_sec > kMaximumGapSeconds) {
    result.invalid_reason = "capture_gap";
  } else {
    std::vector<double> resampled;
    resampled.reserve(kResampleCount);
    std::size_t right_index = 1U;
    bool all_targets_covered = true;
    for (std::size_t index = 0; index < kResampleCount; ++index) {
      const double target_time =
          window_start + static_cast<double>(index) / kResampleFps;
      while (right_index < source.size() &&
             source[right_index].timestamp_sec < target_time) {
        ++right_index;
      }
      if (right_index == source.size()) {
        all_targets_covered = false;
        break;
      }
      const Sample& right = source[right_index];
      const Sample& left = source[right_index - 1U];
      if (left.timestamp_sec > target_time || right.timestamp_sec < target_time) {
        all_targets_covered = false;
        break;
      }
      const double span = right.timestamp_sec - left.timestamp_sec;
      if (span <= 0.0) {
        all_targets_covered = false;
        break;
      }
      const double fraction = (target_time - left.timestamp_sec) / span;
      const double pair_scale = std::max(std::abs(left.green), std::abs(right.green));
      if (!std::isfinite(pair_scale)) {
        all_targets_covered = false;
        break;
      }
      if (pair_scale == 0.0) {
        resampled.push_back(0.0);
        continue;
      }
      const double blended = (left.green / pair_scale) * (1.0 - fraction) +
                             (right.green / pair_scale) * fraction;
      const double interpolated =
          pair_scale * std::clamp(blended, -1.0, 1.0);
      if (!std::isfinite(interpolated)) {
        all_targets_covered = false;
        break;
      }
      resampled.push_back(interpolated);
    }

    if (!all_targets_covered) {
      latest_ = result;
      last_evaluation_timestamp_sec_ = window_end;
      ++evaluation_count_;
      return;
    }

    double resampled_scale = 0.0;
    for (double value : resampled) {
      resampled_scale = std::max(resampled_scale, std::abs(value));
    }
    if (!std::isfinite(resampled_scale)) {
      latest_ = finite_failure_result(result, "low_confidence");
      last_evaluation_timestamp_sec_ = window_end;
      ++evaluation_count_;
      return;
    }
    if (resampled_scale > 0.0) {
      for (double& value : resampled) {
        value /= resampled_scale;
      }
    }

    double mean_time = 0.0;
    double mean_value = 0.0;
    for (std::size_t index = 0; index < resampled.size(); ++index) {
      mean_time += static_cast<double>(index) / kResampleFps;
      mean_value += resampled[index];
    }
    mean_time /= static_cast<double>(resampled.size());
    mean_value /= static_cast<double>(resampled.size());

    double time_variance = 0.0;
    double covariance = 0.0;
    for (std::size_t index = 0; index < resampled.size(); ++index) {
      const double time = static_cast<double>(index) / kResampleFps;
      time_variance += (time - mean_time) * (time - mean_time);
      covariance += (time - mean_time) * (resampled[index] - mean_value);
    }
    const double slope = time_variance > 0.0 ? covariance / time_variance : 0.0;

    std::vector<double> detrended;
    detrended.reserve(resampled.size());
    double maximum_absolute = 0.0;
    for (std::size_t index = 0; index < resampled.size(); ++index) {
      const double time = static_cast<double>(index) / kResampleFps;
      const double value = resampled[index] - mean_value - slope * (time - mean_time);
      detrended.push_back(value);
      maximum_absolute = std::max(maximum_absolute, std::abs(value));
    }
    std::vector<double> normalized_detrended;
    normalized_detrended.reserve(detrended.size());
    result.waveform.reserve(detrended.size());
    for (double value : detrended) {
      const double normalized = maximum_absolute > kEpsilon ? value / maximum_absolute : 0.0;
      normalized_detrended.push_back(normalized);
      result.waveform.push_back(static_cast<float>(normalized));
    }

    double total_power = 0.0;
    double peak_power = 0.0;
    double peak_frequency = 0.0;
    for (int frequency_bin = kMinimumFrequencyBin;
         frequency_bin <= kMaximumFrequencyBin; ++frequency_bin) {
      const double frequency = static_cast<double>(frequency_bin) / 10.0;
      double real = 0.0;
      double imaginary = 0.0;
      for (std::size_t index = 0; index < normalized_detrended.size(); ++index) {
        const double phase = -2.0 * kPi * frequency *
                             static_cast<double>(index) / kResampleFps;
        const double hann = 0.5 - 0.5 * std::cos(
            2.0 * kPi * static_cast<double>(index) /
            static_cast<double>(normalized_detrended.size() - 1U));
        const double weighted = normalized_detrended[index] * hann;
        real += weighted * std::cos(phase);
        imaginary += weighted * std::sin(phase);
      }
      const double power = real * real + imaginary * imaginary;
      total_power += power;
      const bool higher_power = power > peak_power;
      const bool near_equal_higher_frequency =
          frequency > peak_frequency && peak_power > 0.0 &&
          peak_power - power <= peak_power * kPeakTieRelativeTolerance;
      if (higher_power || near_equal_higher_frequency) {
        peak_power = power;
        peak_frequency = frequency;
      }
    }
    result.bpm = peak_frequency * 60.0;
    result.confidence = total_power > kEpsilon ? peak_power / total_power : 0.0;
    if (result.confidence >= kMinimumConfidence && result.bpm >= kMinimumBpm &&
        result.bpm <= kMaximumBpm) {
      result.is_valid = true;
      result.invalid_reason.clear();
    } else {
      result.invalid_reason = "low_confidence";
    }
  }

  if (!finite_result(result)) {
    latest_ = finite_failure_result(result, "low_confidence");
    last_evaluation_timestamp_sec_ = window_end;
    ++evaluation_count_;
    return;
  }

  latest_ = result;
  last_evaluation_timestamp_sec_ = window_end;
  ++evaluation_count_;
}

}  // namespace rppg_qnn
