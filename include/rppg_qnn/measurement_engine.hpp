#pragma once

#include "rppg_qnn/contracts.hpp"

#include <optional>
#include <vector>

namespace rppg_qnn {

class MeasurementEngine {
 public:
  explicit MeasurementEngine(QualityProfile profile = {});

  [[nodiscard]] MeasurementSnapshot evaluate(
      std::vector<CandidateResult> candidates, QualityMetrics quality,
      double window_end_sec);
  void reset();

 private:
  QualityProfile profile_;
  std::optional<double> last_reliable_bpm_;
  std::optional<double> last_reliable_timestamp_sec_;
};

}  // namespace rppg_qnn
