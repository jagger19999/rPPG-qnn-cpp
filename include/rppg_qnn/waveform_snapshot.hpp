#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "rppg_qnn/contracts.hpp"

namespace rppg_qnn {

struct WaveformSnapshot {
  bool available{false};
  std::uint64_t revision{0};
  std::string method;
  double sample_rate_hz{0.0};
  double window_end_sec{0.0};
  bool is_valid{false};
  std::string invalid_reason;
  std::vector<float> values;
};

[[nodiscard]] WaveformSnapshot make_waveform_snapshot(
    const std::vector<float>& values, std::string method,
    double sample_rate_hz, bool is_valid, std::string invalid_reason,
    double window_end_sec = 0.0);

[[nodiscard]] double waveform_sample_rate_hz(const HeartRateResult& result);

}  // namespace rppg_qnn
