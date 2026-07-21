#pragma once

#include "rppg_qnn/contracts.hpp"

#include <cstddef>
#include <deque>
#include <optional>

#include <opencv2/core.hpp>

namespace rppg_qnn {

class GreenPredictor {
 public:
  void add_sample(double timestamp_sec, const cv::Scalar& mean_bgr);

  [[nodiscard]] std::optional<HeartRateResult> latest_result() const;
  [[nodiscard]] std::size_t buffered_count() const;
  [[nodiscard]] double buffered_span_sec() const;
  [[nodiscard]] std::size_t evaluation_count() const;
  void reset();

 private:
  struct Sample {
    double timestamp_sec;
    double green;
  };

  void set_sampling_result();
  void evaluate();

  std::deque<Sample> samples_;
  std::optional<HeartRateResult> latest_;
  std::optional<double> last_evaluation_timestamp_sec_;
  std::size_t evaluation_count_{0};
};

}  // namespace rppg_qnn
