#include "rppg_qnn/deep_stabilizer.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <utility>
#include <vector>

namespace rppg_qnn {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSampleRateHz = 30.0;
constexpr double kMinimumBpm = 45.0;
constexpr double kMaximumBpm = 150.0;
constexpr double kMinimumConfidence = 0.10;

// These are conservative structural gates, not values fitted to a live run.
// Synthetic two-tone tests lock their meaning: an octave candidate must carry
// at least half the raw peak power, while a large new peak must dominate the
// old rate by 1.5x before temporal history is discarded.
constexpr double kHarmonicSupportRatio = 0.50;
constexpr double kLargeJumpBpm = 20.0;
constexpr double kJumpDominanceRatio = 1.50;
constexpr double kHarmonicReferenceToleranceBpm = 12.0;
constexpr double kConstantRangeEpsilon = 1e-6;

DeepStabilityResult rejected(const char* reason) {
  return {0.0, false, reason};
}

double median(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  const std::size_t middle = values.size() / 2U;
  if ((values.size() % 2U) != 0U) {
    return values[middle];
  }
  return (values[middle - 1U] + values[middle]) / 2.0;
}

double spectral_power(const std::vector<float>& waveform, double bpm) {
  if (bpm < kMinimumBpm || bpm > kMaximumBpm) {
    return 0.0;
  }
  const double mean =
      std::accumulate(waveform.begin(), waveform.end(), 0.0) /
      static_cast<double>(waveform.size());
  double real = 0.0;
  double imaginary = 0.0;
  for (std::size_t sample = 0; sample < waveform.size(); ++sample) {
    const double window =
        0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(sample) /
                             static_cast<double>(waveform.size() - 1U));
    const double angle =
        2.0 * kPi * (bpm / 60.0) * static_cast<double>(sample) /
        kSampleRateHz;
    const double value =
        (static_cast<double>(waveform[sample]) - mean) * window;
    real += value * std::cos(angle);
    imaginary -= value * std::sin(angle);
  }
  return real * real + imaginary * imaginary;
}

bool has_support(double candidate_power, double reference_power,
                 double minimum_ratio) {
  return std::isfinite(candidate_power) && std::isfinite(reference_power) &&
         reference_power > 0.0 &&
         candidate_power / reference_power >= minimum_ratio;
}

}  // namespace

DeepStabilityResult DeepStabilizer::stabilize(
    const std::vector<float>& waveform, double raw_bpm, double confidence) {
  if (waveform.size() < 2U ||
      !std::all_of(waveform.begin(), waveform.end(),
                   [](float value) { return std::isfinite(value); })) {
    return rejected("rejected_nonfinite_waveform");
  }
  const auto [minimum, maximum] =
      std::minmax_element(waveform.begin(), waveform.end());
  if (static_cast<double>(*maximum) - static_cast<double>(*minimum) <=
      kConstantRangeEpsilon) {
    return rejected("rejected_constant_waveform");
  }
  if (!std::isfinite(confidence) || confidence < kMinimumConfidence) {
    return rejected("rejected_low_confidence");
  }
  if (!std::isfinite(raw_bpm) || raw_bpm < kMinimumBpm ||
      raw_bpm > kMaximumBpm) {
    return rejected("rejected_invalid_bpm");
  }

  double accepted_bpm = raw_bpm;
  const char* reason = "accepted_raw";
  const double raw_power = spectral_power(waveform, raw_bpm);
  if (!accepted_bpm_.empty()) {
    const double reference =
        median(std::vector<double>(accepted_bpm_.begin(), accepted_bpm_.end()));
    const std::pair<double, const char*> harmonic_candidates[] = {
        {raw_bpm / 2.0, "corrected_to_half"},
        {raw_bpm * 2.0, "corrected_to_double"}};
    for (const auto& [candidate, candidate_reason] : harmonic_candidates) {
      if (candidate >= kMinimumBpm && candidate <= kMaximumBpm &&
          std::abs(candidate - reference) <=
              kHarmonicReferenceToleranceBpm &&
          has_support(spectral_power(waveform, candidate), raw_power,
                      kHarmonicSupportRatio)) {
        accepted_bpm = candidate;
        reason = candidate_reason;
        break;
      }
    }

    if (std::abs(accepted_bpm - reference) > kLargeJumpBpm) {
      const double prior_power = spectral_power(waveform, reference);
      const double accepted_power = spectral_power(waveform, accepted_bpm);
      if (!has_support(accepted_power, prior_power, kJumpDominanceRatio)) {
        return rejected("rejected_unsupported_jump");
      }
      accepted_bpm_.clear();
      reason = "accepted_supported_jump";
    }
  }

  accepted_bpm_.push_back(accepted_bpm);
  while (accepted_bpm_.size() > 3U) {
    accepted_bpm_.pop_front();
  }
  const double display =
      median(std::vector<double>(accepted_bpm_.begin(), accepted_bpm_.end()));
  return {display, true, reason};
}

void DeepStabilizer::reset() noexcept { accepted_bpm_.clear(); }

}  // namespace rppg_qnn
