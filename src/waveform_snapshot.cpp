#include "rppg_qnn/waveform_snapshot.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <utility>

namespace rppg_qnn {

double waveform_sample_rate_hz(const HeartRateResult& result) {
  if (result.method == "TSCAN" || result.method == "DEEP") {
    return 30.0;
  }
  const double span = result.window_end_sec - result.window_start_sec;
  if (result.waveform.size() >= 2U && std::isfinite(span) && span > 0.0) {
    return static_cast<double>(result.waveform.size() - 1U) / span;
  }
  return result.source_fps > 0.0 ? result.source_fps : 30.0;
}

WaveformSnapshot make_waveform_snapshot(const std::vector<float>& values,
                                        std::string method,
                                        double sample_rate_hz, bool is_valid,
                                        std::string invalid_reason) {
  WaveformSnapshot snapshot;
  if (values.size() < 2U || !std::isfinite(sample_rate_hz) ||
      sample_rate_hz <= 0.0 ||
      !std::all_of(values.begin(), values.end(),
                   [](float value) { return std::isfinite(value); })) {
    return snapshot;
  }
  const double mean = std::accumulate(values.begin(), values.end(), 0.0) /
                      static_cast<double>(values.size());
  double scale = 0.0;
  for (float value : values) {
    scale = std::max(scale, std::abs(static_cast<double>(value) - mean));
  }
  if (!std::isfinite(scale) || scale <= 1e-12) {
    return snapshot;
  }
  snapshot.values.reserve(values.size());
  for (float value : values) {
    const double normalized = (static_cast<double>(value) - mean) / scale;
    snapshot.values.push_back(
        static_cast<float>(std::clamp(normalized, -1.0, 1.0)));
  }
  snapshot.available = true;
  snapshot.method = std::move(method);
  snapshot.sample_rate_hz = sample_rate_hz;
  snapshot.is_valid = is_valid;
  snapshot.invalid_reason = std::move(invalid_reason);
  return snapshot;
}

}  // namespace rppg_qnn
