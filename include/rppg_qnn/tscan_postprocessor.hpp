#pragma once

#include <string>
#include <vector>

namespace rppg_qnn {

struct TscanPostprocessResult {
  double bpm{0.0};
  double confidence{0.0};
  bool is_valid{false};
  std::string invalid_reason;
};

TscanPostprocessResult postprocess_tscan_waveform(
    const std::vector<float>& waveform, double confidence_threshold = 0.10);

}  // namespace rppg_qnn
