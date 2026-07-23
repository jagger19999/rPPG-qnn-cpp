#pragma once

#include "rppg_qnn/contracts.hpp"

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rppg_qnn {

enum class TraditionalMethod { Green, Pos, Chrom };

[[nodiscard]] TraditionalMethod traditional_method_from_string(
    const std::string& method);
[[nodiscard]] std::string traditional_method_name(TraditionalMethod method);

// Input samples are RGB channel means at 30 FPS. The returned trace has the
// same length and follows the Python project's traditional.py contract.
[[nodiscard]] std::vector<double> extract_traditional_bvp(
    const std::vector<cv::Vec3d>& rgb,
    TraditionalMethod method);

class TraditionalPredictor {
 public:
  explicit TraditionalPredictor(
      TraditionalMethod method = TraditionalMethod::Green);

  void add_sample(double timestamp_sec, const cv::Scalar& mean_bgr);

  [[nodiscard]] std::optional<HeartRateResult> latest_result() const;
  [[nodiscard]] std::size_t buffered_count() const;
  [[nodiscard]] double buffered_span_sec() const;
  [[nodiscard]] std::size_t evaluation_count() const;
  [[nodiscard]] TraditionalMethod method() const;
  void reset();

 private:
  struct Sample {
    double timestamp_sec;
    cv::Vec3d rgb;
  };

  void set_sampling_result();
  void evaluate();

  TraditionalMethod method_;
  std::deque<Sample> samples_;
  std::optional<HeartRateResult> latest_;
  std::optional<double> last_evaluation_timestamp_sec_;
  std::size_t evaluation_count_{0};
};

}  // namespace rppg_qnn
