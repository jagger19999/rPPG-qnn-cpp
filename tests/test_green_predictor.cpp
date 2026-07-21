#include "rppg_qnn/green_predictor.hpp"

#include <cmath>
#include <cstddef>
#include <limits>
#include <string>

#include "test_support.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;

void add_sine_samples(rppg_qnn::GreenPredictor& predictor,
                      double duration_sec,
                      double fps,
                      bool jitter = false) {
  double timestamp = 0.0;
  std::size_t index = 0;
  while (timestamp <= duration_sec) {
    const double green = 100.0 + 2.0 * std::sin(2.0 * kPi * 1.2 * timestamp);
    predictor.add_sample(timestamp, cv::Scalar(30.0, green, 40.0));
    const double phase = static_cast<double>(index % 7U) - 3.0;
    const double interval = (1.0 / fps) * (jitter ? 1.0 + 0.06 * phase / 3.0 : 1.0);
    timestamp += interval;
    ++index;
  }
}

void expect_reason(const rppg_qnn::GreenPredictor& predictor,
                   const std::string& expected_reason) {
  const auto result = predictor.latest_result();
  EXPECT_TRUE(result.has_value());
  if (result.has_value()) {
    EXPECT_EQ(result->is_valid, false);
    EXPECT_EQ(result->invalid_reason, expected_reason);
  }
}

void valid_jittered_signal_estimates_72_bpm() {
  rppg_qnn::GreenPredictor predictor;
  add_sine_samples(predictor, 20.0, 29.0, true);

  const auto result = predictor.latest_result();
  EXPECT_TRUE(result.has_value());
  if (!result.has_value()) {
    return;
  }
  EXPECT_EQ(result->method, "GREEN");
  EXPECT_TRUE(result->is_valid);
  EXPECT_TRUE(std::abs(result->bpm - 72.0) <= 2.0);
  EXPECT_TRUE(result->source_fps >= 27.0 && result->source_fps <= 31.0);
  EXPECT_TRUE(!result->waveform.empty());
  for (float value : result->waveform) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

void short_stream_reports_sampling() {
  rppg_qnn::GreenPredictor predictor;
  add_sine_samples(predictor, 7.9, 30.0);
  expect_reason(predictor, "sampling");
}

void low_source_fps_is_rejected() {
  rppg_qnn::GreenPredictor predictor;
  add_sine_samples(predictor, 12.0, 10.0);
  expect_reason(predictor, "low_source_fps");
}

void capture_gap_is_rejected_at_high_average_fps() {
  rppg_qnn::GreenPredictor predictor;
  for (int index = 0; index <= 300; ++index) {
    const double timestamp = static_cast<double>(index) / 30.0;
    if (timestamp > 5.0 && timestamp < 6.0) {
      continue;
    }
    const double green = 100.0 + 2.0 * std::sin(2.0 * kPi * 1.2 * timestamp);
    predictor.add_sample(timestamp, cv::Scalar(30.0, green, 40.0));
  }
  expect_reason(predictor, "capture_gap");
}

void flat_signal_is_low_confidence() {
  rppg_qnn::GreenPredictor predictor;
  for (int index = 0; index <= 360; ++index) {
    const double timestamp = static_cast<double>(index) / 30.0;
    predictor.add_sample(timestamp, cv::Scalar(30.0, 100.0, 40.0));
  }
  expect_reason(predictor, "low_confidence");
}

void invalid_input_resets_safely_and_reports_sampling() {
  rppg_qnn::GreenPredictor predictor;
  add_sine_samples(predictor, 10.0, 30.0);
  predictor.add_sample(9.0, cv::Scalar(30.0, 100.0, 40.0));
  expect_reason(predictor, "sampling");
  predictor.add_sample(10.1, cv::Scalar(30.0, std::numeric_limits<double>::quiet_NaN(), 40.0));
  expect_reason(predictor, "sampling");
  EXPECT_EQ(predictor.buffered_count(), std::size_t{0});
}

void long_stream_is_bounded_and_evaluations_are_throttled() {
  rppg_qnn::GreenPredictor predictor;
  add_sine_samples(predictor, 45.0, 30.0);
  EXPECT_TRUE(predictor.buffered_count() <= 901U);
  const std::size_t evaluations_before = predictor.evaluation_count();
  const auto first = predictor.latest_result();
  predictor.add_sample(45.01, cv::Scalar(30.0, 100.0, 40.0));
  const auto second = predictor.latest_result();
  EXPECT_EQ(predictor.evaluation_count(), evaluations_before);
  EXPECT_TRUE(first.has_value() && second.has_value());
}

}  // namespace

int main() {
  valid_jittered_signal_estimates_72_bpm();
  short_stream_reports_sampling();
  low_source_fps_is_rejected();
  capture_gap_is_rejected_at_high_average_fps();
  flat_signal_is_low_confidence();
  invalid_input_resets_safely_and_reports_sampling();
  long_stream_is_bounded_and_evaluations_are_throttled();
  return test_support::finish();
}
