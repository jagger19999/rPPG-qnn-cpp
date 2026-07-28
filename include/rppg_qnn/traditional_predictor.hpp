#pragma once

#include "rppg_qnn/contracts.hpp"

#include <array>
#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rppg_qnn {

enum class TraditionalMethod { Green, Pos, Chrom, Lgi };

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

  // Enable/disable exponential spectral smoothing (default: enabled, α=0.85).
  // When enabled, power at each frequency bin is smoothed across evaluations:
  //   smoothed_power[f] = α * smoothed_power[f] + (1-α) * raw_power[f]
  // This stabilises the selected BPM and reduces jump-to-jump variance.
  void set_spectral_smoothing(bool enabled, double alpha = 0.85);

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

  // ES spectral smoothing (Python ExponentialSpectrumSmoother equivalent)
  bool spectral_smoothing_enabled_{true};
  double spectral_smoothing_alpha_{0.85};
  static constexpr int kMinimumBin = 7;
  static constexpr int kMaximumBin = 30;
  std::array<double, kMaximumBin - kMinimumBin + 1> smoothed_power_{};
  bool smoothing_initialized_{false};
};

}  // namespace rppg_qnn
