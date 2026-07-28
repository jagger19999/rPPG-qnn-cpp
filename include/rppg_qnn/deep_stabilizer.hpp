#pragma once

#include <deque>
#include <string>
#include <vector>

namespace rppg_qnn {

struct DeepStabilityResult {
  double display_bpm{0.0};
  bool stability_valid{false};
  std::string correction_reason;
};

class DeepStabilizer {
 public:
  [[nodiscard]] DeepStabilityResult stabilize(
      const std::vector<float>& waveform, double raw_bpm, double confidence);
  void reset() noexcept;

 private:
  std::deque<double> accepted_bpm_;
};

}  // namespace rppg_qnn
