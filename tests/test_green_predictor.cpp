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
    EXPECT_TRUE(std::isfinite(result->window_start_sec));
    EXPECT_TRUE(std::isfinite(result->window_end_sec));
    EXPECT_TRUE(std::isfinite(result->bpm));
    EXPECT_TRUE(std::isfinite(result->confidence));
    EXPECT_TRUE(std::isfinite(result->source_fps));
    EXPECT_TRUE(std::isfinite(result->max_frame_gap_sec));
    for (float value : result->waveform) {
      EXPECT_TRUE(std::isfinite(value));
    }
  }
}

double green_signal(double timestamp) {
  return 100.0 + 2.0 * std::sin(2.0 * kPi * 1.2 * timestamp);
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
  EXPECT_EQ(result->backend, "cpu");
  EXPECT_TRUE(result->is_valid);
  EXPECT_TRUE(std::abs(result->bpm - 72.0) <= 2.0);
  EXPECT_TRUE(std::abs((result->window_end_sec - result->window_start_sec) - 10.0) <=
              1e-9);
  EXPECT_TRUE(result->confidence >= 0.10);
  EXPECT_TRUE(result->source_fps >= 27.0 && result->source_fps <= 31.0);
  EXPECT_TRUE(result->source_frame_count >= 270U && result->source_frame_count <= 310U);
  EXPECT_TRUE(result->max_frame_gap_sec < 0.75);
  EXPECT_EQ(result->waveform.size(), std::size_t{300});
  double waveform_sum = 0.0;
  for (float value : result->waveform) {
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_TRUE(value >= -1.0F && value <= 1.0F);
    waveform_sum += value;
  }
  EXPECT_TRUE(std::abs(waveform_sum / static_cast<double>(result->waveform.size())) < 0.01);
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
    predictor.add_sample(timestamp, cv::Scalar(30.0, green_signal(timestamp), 40.0));
  }
  expect_reason(predictor, "capture_gap");
}

void gap_crossing_window_start_is_rejected() {
  rppg_qnn::GreenPredictor predictor;
  for (int index = 0; index <= 450; ++index) {
    const double timestamp = static_cast<double>(index) / 30.0;
    if (timestamp > 4.8 && timestamp < 5.8) {
      continue;
    }
    predictor.add_sample(timestamp, cv::Scalar(30.0, green_signal(timestamp), 40.0));
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

void nonmonotonic_input_resets_safely_and_reports_sampling() {
  rppg_qnn::GreenPredictor predictor;
  add_sine_samples(predictor, 10.0, 30.0);
  predictor.add_sample(9.0, cv::Scalar(30.0, 100.0, 40.0));
  expect_reason(predictor, "sampling");
  EXPECT_EQ(predictor.buffered_count(), std::size_t{0});
}

void nan_timestamp_resets_safely_and_reports_sampling() {
  rppg_qnn::GreenPredictor predictor;
  add_sine_samples(predictor, 10.0, 30.0);
  predictor.add_sample(std::numeric_limits<double>::quiet_NaN(),
                       cv::Scalar(30.0, 100.0, 40.0));
  expect_reason(predictor, "sampling");
  EXPECT_EQ(predictor.buffered_count(), std::size_t{0});
}

void nan_bgr_resets_safely_and_reports_sampling() {
  rppg_qnn::GreenPredictor predictor;
  add_sine_samples(predictor, 10.0, 30.0);
  predictor.add_sample(10.1, cv::Scalar(30.0, std::numeric_limits<double>::quiet_NaN(), 40.0));
  expect_reason(predictor, "sampling");
  EXPECT_EQ(predictor.buffered_count(), std::size_t{0});
}

void evaluation_is_throttled_after_the_first_full_window() {
  rppg_qnn::GreenPredictor predictor;
  for (int index = 0; index <= 300; ++index) {
    const double timestamp = static_cast<double>(index) / 30.0;
    predictor.add_sample(timestamp, cv::Scalar(30.0, green_signal(timestamp), 40.0));
  }
  EXPECT_EQ(predictor.evaluation_count(), std::size_t{1});
  predictor.add_sample(10.5, cv::Scalar(30.0, green_signal(10.5), 40.0));
  predictor.add_sample(10.9, cv::Scalar(30.0, green_signal(10.9), 40.0));
  EXPECT_EQ(predictor.evaluation_count(), std::size_t{1});
  predictor.add_sample(11.0, cv::Scalar(30.0, green_signal(11.0), 40.0));
  EXPECT_EQ(predictor.evaluation_count(), std::size_t{2});
}

void history_is_bounded_to_thirty_seconds() {
  rppg_qnn::GreenPredictor predictor;
  for (int index = 0; index <= 1350; ++index) {
    const double timestamp = static_cast<double>(index) / 30.0;
    predictor.add_sample(timestamp, cv::Scalar(30.0, green_signal(timestamp), 40.0));
  }
  EXPECT_TRUE(predictor.buffered_span_sec() <= 30.0);
  EXPECT_TRUE(predictor.buffered_span_sec() >= 29.9);
}

void public_reset_clears_result_and_evaluation_state() {
  rppg_qnn::GreenPredictor predictor;
  for (int index = 0; index <= 300; ++index) {
    const double timestamp = static_cast<double>(index) / 30.0;
    predictor.add_sample(timestamp, cv::Scalar(30.0, green_signal(timestamp), 40.0));
  }
  predictor.reset();
  EXPECT_TRUE(!predictor.latest_result().has_value());
  EXPECT_EQ(predictor.buffered_count(), std::size_t{0});
  EXPECT_EQ(predictor.evaluation_count(), std::size_t{0});
  predictor.add_sample(0.0, cv::Scalar(30.0, green_signal(0.0), 40.0));
  expect_reason(predictor, "sampling");
}

}  // namespace

int main() {
  valid_jittered_signal_estimates_72_bpm();
  short_stream_reports_sampling();
  low_source_fps_is_rejected();
  capture_gap_is_rejected_at_high_average_fps();
  gap_crossing_window_start_is_rejected();
  flat_signal_is_low_confidence();
  nonmonotonic_input_resets_safely_and_reports_sampling();
  nan_timestamp_resets_safely_and_reports_sampling();
  nan_bgr_resets_safely_and_reports_sampling();
  evaluation_is_throttled_after_the_first_full_window();
  history_is_bounded_to_thirty_seconds();
  public_reset_clears_result_and_evaluation_state();
  return test_support::finish();
}
