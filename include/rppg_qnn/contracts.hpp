#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace rppg_qnn {

struct HeartRateResult {
  int schema_version{1};
  std::string method;
  double window_start_sec{0.0};
  double window_end_sec{0.0};
  double bpm{0.0};
  double confidence{0.0};
  bool is_valid{false};
  std::string invalid_reason;
  double source_fps{0.0};
  std::size_t source_frame_count{0};
  double max_frame_gap_sec{0.0};
  double inference_ms{0.0};
  std::string backend;
  std::string model_sha256;
  std::vector<float> waveform;
};

}  // namespace rppg_qnn
