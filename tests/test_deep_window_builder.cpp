#include "rppg_qnn/deep_window_builder.hpp"

#include "test_support.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>

namespace {

rppg_qnn::RoiPacket packet(double timestamp, const cv::Scalar& bgr = {3, 17, 251}) {
  cv::Mat roi(72, 72, CV_8UC3, bgr);
  return rppg_qnn::RoiPacket{0, timestamp, roi, std::nullopt, false};
}

void builds_rgb_nhwc_window_from_jittered_capture() {
  rppg_qnn::DeepWindowBuilder builder(6.0, 180, {72, 72});
  std::optional<rppg_qnn::DeepInput> output;
  double timestamp = 0.0;
  for (int index = 0; timestamp <= 7.1; ++index) {
    output = builder.add_roi(packet(timestamp));
    timestamp += 1.0 / static_cast<double>(27 + index % 5);
  }

  EXPECT_TRUE(output.has_value());
  if (!output.has_value()) {
    return;
  }
  EXPECT_EQ(output->shape, (std::vector<std::int64_t>{180, 72, 72, 3}));
  EXPECT_TRUE(std::abs((output->end_sec - output->start_sec) - 6.0) < 1e-9);
  EXPECT_TRUE(output->source_fps >= 27.0 && output->source_fps <= 31.5);
  EXPECT_TRUE(output->max_frame_gap_sec < 0.75);
  EXPECT_EQ(output->tensor[0], 251.0F);
  EXPECT_EQ(output->tensor[1], 17.0F);
  EXPECT_EQ(output->tensor[2], 3.0F);
  for (float value : output->tensor) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

void reports_capture_quality_and_rejects_invalid_rois() {
  rppg_qnn::DeepWindowBuilder sparse(6.0, 180, {72, 72});
  for (int index = 0; index <= 6; ++index) {
    (void)sparse.add_roi(packet(static_cast<double>(index)));
  }
  EXPECT_EQ(sparse.status(), std::string("low_source_fps"));

  rppg_qnn::DeepWindowBuilder gapped(6.0, 180, {72, 72});
  double gap_timestamp = 0.0;
  bool skipped_second = false;
  while (gap_timestamp <= 7.0) {
    (void)gapped.add_roi(packet(gap_timestamp));
    gap_timestamp += 1.0 / 30.0;
    if (!skipped_second && gap_timestamp >= 2.0) {
      gap_timestamp += 1.0;
      skipped_second = true;
    }
  }
  EXPECT_EQ(gapped.status(), std::string("capture_gap"));

  rppg_qnn::DeepWindowBuilder builder(6.0, 180, {72, 72});
  EXPECT_TRUE(!builder.add_roi(packet(1.0)).has_value());
  EXPECT_TRUE(!builder.add_roi(packet(0.9)).has_value());
  EXPECT_EQ(builder.status(), std::string("nonmonotonic_timestamp"));
  rppg_qnn::RoiPacket empty{0, 2.0, {}, std::nullopt, false};
  EXPECT_TRUE(!builder.add_roi(empty).has_value());
  EXPECT_EQ(builder.status(), std::string("empty_roi"));

  rppg_qnn::DeepWindowBuilder invalid_format(6.0, 180, {72, 72});
  for (int index = 0; index <= 180; ++index) {
    (void)invalid_format.add_roi(packet(static_cast<double>(index) / 30.0));
  }
  cv::Mat float_roi(72, 72, CV_32FC3, cv::Scalar(1.0, 2.0, 3.0));
  rppg_qnn::RoiPacket malformed{0, 6.1, float_roi, std::nullopt, false};
  EXPECT_TRUE(!invalid_format.add_roi(malformed).has_value());
  EXPECT_EQ(invalid_format.status(), std::string("invalid_roi_format"));

  bool rejected_oversized_output = false;
  try {
    (void)rppg_qnn::DeepWindowBuilder(6.0, 180, {5000, 72});
  } catch (const std::invalid_argument&) {
    rejected_oversized_output = true;
  }
  EXPECT_TRUE(rejected_oversized_output);
}

void copies_roi_pixels_before_the_source_is_reused() {
  rppg_qnn::DeepWindowBuilder builder(6.0, 180, {72, 72});
  cv::Mat roi(72, 72, CV_8UC3, cv::Scalar(9, 8, 7));
  for (int index = 0; index <= 180; ++index) {
    const double timestamp = static_cast<double>(index) / 30.0;
    auto source = rppg_qnn::RoiPacket{0, timestamp, roi, std::nullopt, false};
    const auto value = builder.add_roi(source);
    roi.setTo(cv::Scalar(1, 2, 3));
    if (value.has_value()) {
      EXPECT_EQ(value->tensor[0], 7.0F);
      EXPECT_EQ(value->tensor[1], 8.0F);
      EXPECT_EQ(value->tensor[2], 9.0F);
    }
    roi.setTo(cv::Scalar(9, 8, 7));
  }
}

}  // namespace

int main() {
  builds_rgb_nhwc_window_from_jittered_capture();
  reports_capture_quality_and_rejects_invalid_rois();
  copies_roi_pixels_before_the_source_is_reused();
  return test_support::finish();
}
