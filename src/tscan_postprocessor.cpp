#include "rppg_qnn/tscan_postprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace rppg_qnn {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kFps = 30.0;
constexpr double kMinimumFrequency = 0.75;
constexpr double kMaximumFrequency = 2.5;

TscanPostprocessResult invalid_waveform() {
  TscanPostprocessResult result;
  result.invalid_reason = "TSCAN_WAVEFORM_INVALID";
  return result;
}

}  // namespace

TscanPostprocessResult postprocess_tscan_waveform(
    const std::vector<float>& waveform, double confidence_threshold) {
  if (!std::isfinite(confidence_threshold) || confidence_threshold < 0.0 ||
      confidence_threshold > 1.0) {
    throw std::invalid_argument("confidence_threshold must be finite and in [0, 1]");
  }
  if (waveform.size() < 8U ||
      !std::all_of(waveform.begin(), waveform.end(),
                   [](float value) { return std::isfinite(value); })) {
    return invalid_waveform();
  }

  const std::size_t sample_count = waveform.size();
  const double mean =
      std::accumulate(waveform.begin(), waveform.end(), 0.0) /
      static_cast<double>(sample_count);

  std::vector<double> windowed(sample_count);
  for (std::size_t sample = 0; sample < sample_count; ++sample) {
    const double window =
        0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(sample) /
                             static_cast<double>(sample_count - 1U));
    windowed[sample] = (static_cast<double>(waveform[sample]) - mean) * window;
  }

  std::vector<double> band_power;
  std::vector<double> band_frequency;
  for (std::size_t bin = 0; bin <= sample_count / 2U; ++bin) {
    const double frequency =
        static_cast<double>(bin) * kFps / static_cast<double>(sample_count);
    if (frequency < kMinimumFrequency || frequency > kMaximumFrequency) {
      continue;
    }

    double real = 0.0;
    double imaginary = 0.0;
    for (std::size_t sample = 0; sample < sample_count; ++sample) {
      const double angle = 2.0 * kPi * static_cast<double>(bin) *
                           static_cast<double>(sample) /
                           static_cast<double>(sample_count);
      real += windowed[sample] * std::cos(angle);
      imaginary -= windowed[sample] * std::sin(angle);
    }
    band_frequency.push_back(frequency);
    band_power.push_back(real * real + imaginary * imaginary);
  }

  if (band_power.empty()) {
    return invalid_waveform();
  }

  const auto peak = std::max_element(band_power.begin(), band_power.end());
  const std::size_t peak_index =
      static_cast<std::size_t>(std::distance(band_power.begin(), peak));
  const double total_power =
      std::accumulate(band_power.begin(), band_power.end(), 0.0);
  if (total_power == 0.0 || !std::isfinite(total_power)) {
    return invalid_waveform();
  }

  const double confidence = *peak / total_power;
  if (confidence < confidence_threshold) {
    TscanPostprocessResult result;
    result.confidence = confidence;
    result.invalid_reason = "TSCAN_LOW_CONFIDENCE";
    return result;
  }

  TscanPostprocessResult result;
  result.bpm = band_frequency[peak_index] * 60.0;
  result.confidence = confidence;
  result.is_valid = true;
  return result;
}

}  // namespace rppg_qnn
