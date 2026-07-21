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
constexpr double kMinimumFrequencyHz = 0.7;
constexpr double kMaximumFrequencyHz = 3.0;
constexpr double kFrequencyStepHz = 0.1;
constexpr double kMinimumConfidence = 0.10;
constexpr double kMinimumBpm = 42.0;
constexpr double kMaximumBpm = 180.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-12;

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

std::size_t GreenPredictor::evaluation_count() const { return evaluation_count_; }

void GreenPredictor::reset() {
  samples_.clear();
  last_evaluation_timestamp_sec_.reset();
}

void GreenPredictor::set_sampling_result() { latest_ = invalid_result("sampling"); }

void GreenPredictor::evaluate() {
  const double window_end = samples_.back().timestamp_sec;
  const double window_start = window_end - kWindowSeconds;
  std::vector<Sample> source;
  source.reserve(samples_.size());
  for (const Sample& sample : samples_) {
    if (sample.timestamp_sec >= window_start) {
      source.push_back(sample);
    }
  }

  HeartRateResult result = invalid_result("sampling");
  result.window_start_sec = window_start;
  result.window_end_sec = window_end;
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
    for (std::size_t index = 0; index < kResampleCount; ++index) {
      const double target_time =
          window_start + static_cast<double>(index) / kResampleFps;
      while (right_index < source.size() &&
             source[right_index].timestamp_sec < target_time) {
        ++right_index;
      }
      if (right_index == source.size()) {
        right_index = source.size() - 1U;
      }
      const Sample& right = source[right_index];
      const Sample& left = source[right_index - 1U];
      const double span = right.timestamp_sec - left.timestamp_sec;
      const double fraction = span > 0.0
                                  ? (target_time - left.timestamp_sec) / span
                                  : 0.0;
      resampled.push_back(left.green + fraction * (right.green - left.green));
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
    result.waveform.reserve(detrended.size());
    for (double value : detrended) {
      const double normalized = maximum_absolute > kEpsilon ? value / maximum_absolute : 0.0;
      result.waveform.push_back(static_cast<float>(normalized));
    }

    double total_power = 0.0;
    double peak_power = 0.0;
    double peak_frequency = 0.0;
    for (double frequency = kMinimumFrequencyHz;
         frequency <= kMaximumFrequencyHz + kEpsilon;
         frequency += kFrequencyStepHz) {
      double real = 0.0;
      double imaginary = 0.0;
      for (std::size_t index = 0; index < detrended.size(); ++index) {
        const double phase = -2.0 * kPi * frequency *
                             static_cast<double>(index) / kResampleFps;
        const double hann = 0.5 - 0.5 * std::cos(
            2.0 * kPi * static_cast<double>(index) /
            static_cast<double>(detrended.size() - 1U));
        const double weighted = detrended[index] * hann;
        real += weighted * std::cos(phase);
        imaginary += weighted * std::sin(phase);
      }
      const double power = real * real + imaginary * imaginary;
      total_power += power;
      if (power > peak_power) {
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

  latest_ = result;
  last_evaluation_timestamp_sec_ = window_end;
  ++evaluation_count_;
}

}  // namespace rppg_qnn
