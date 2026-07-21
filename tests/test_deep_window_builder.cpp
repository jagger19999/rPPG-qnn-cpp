#include "rppg_qnn/deep_window_builder.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

struct CapturedFrame {
  double timestamp_sec;
  int id;
};

cv::Vec3b bgr_at(int id, int row, int column) {
  return {static_cast<unsigned char>((id * 3 + column) % 251),
          static_cast<unsigned char>(50 + (id * 5 + row) % 150),
          static_cast<unsigned char>(100 + (id * 7 + row + column) % 120)};
}

rppg_qnn::RoiPacket patterned_packet(double timestamp, int id) {
  cv::Mat roi(72, 72, CV_8UC3);
  for (int row = 0; row < roi.rows; ++row) {
    for (int column = 0; column < roi.cols; ++column) {
      roi.at<cv::Vec3b>(row, column) = bgr_at(id, row, column);
    }
  }
  return rppg_qnn::RoiPacket{0, timestamp, roi, std::nullopt, false};
}

rppg_qnn::RoiPacket flat_packet(double timestamp,
                                 const cv::Scalar& bgr = {3, 17, 251}) {
  cv::Mat roi(72, 72, CV_8UC3, bgr);
  return rppg_qnn::RoiPacket{0, timestamp, roi, std::nullopt, false};
}

std::vector<CapturedFrame> supported_frames(const std::vector<CapturedFrame>& frames,
                                            double start, double end) {
  std::vector<CapturedFrame> source;
  for (const CapturedFrame& frame : frames) {
    if (frame.timestamp_sec <= start) {
      source = {frame};
    } else if (frame.timestamp_sec <= end) {
      source.push_back(frame);
    }
  }
  return source;
}

int nearest_id(const std::vector<CapturedFrame>& source, double target) {
  const auto closest = std::min_element(
      source.begin(), source.end(), [target](const CapturedFrame& left,
                                              const CapturedFrame& right) {
        return std::abs(left.timestamp_sec - target) <
               std::abs(right.timestamp_sec - target);
      });
  return closest->id;
}

void expect_rgb_at(const rppg_qnn::DeepInput& input, std::size_t sample,
                   int expected_id, int row, int column) {
  const std::size_t values_per_frame = 72U * 72U * 3U;
  const std::size_t offset = sample * values_per_frame +
                             (static_cast<std::size_t>(row) * 72U +
                              static_cast<std::size_t>(column)) * 3U;
  const cv::Vec3b expected = bgr_at(expected_id, row, column);
  EXPECT_EQ(input.tensor[offset], static_cast<float>(expected[2]));
  EXPECT_EQ(input.tensor[offset + 1U], static_cast<float>(expected[1]));
  EXPECT_EQ(input.tensor[offset + 2U], static_cast<float>(expected[0]));
}

void builds_first_ready_rgb_nhwc_window_from_jittered_capture() {
  rppg_qnn::DeepWindowBuilder builder(6.0, 180, {72, 72});
  std::vector<CapturedFrame> frames;
  std::optional<rppg_qnn::DeepInput> output;
  int ready_count = 0;
  double timestamp = 0.0;
  for (int index = 0; timestamp <= 7.0; ++index) {
    frames.push_back({timestamp, index});
    output = builder.add_roi(patterned_packet(timestamp, index));
    if (output.has_value()) {
      ++ready_count;
      break;
    }
    timestamp += 1.0 / static_cast<double>(27 + index % 5);
  }

  EXPECT_EQ(ready_count, 1);
  EXPECT_TRUE(output.has_value());
  if (!output.has_value()) {
    return;
  }
  const double start = output->end_sec - 6.0;
  const std::vector<CapturedFrame> source =
      supported_frames(frames, start, output->end_sec);
  EXPECT_TRUE(!source.empty());
  EXPECT_EQ(output->shape, (std::vector<std::int64_t>{180, 72, 72, 3}));
  EXPECT_EQ(output->end_sec, frames.back().timestamp_sec);
  EXPECT_TRUE(std::abs(output->start_sec - start) < 1e-12);
  EXPECT_TRUE(std::abs((output->end_sec - output->start_sec) - 6.0) < 1e-12);
  EXPECT_EQ(output->source_frame_count, source.size());
  const double expected_fps = static_cast<double>(source.size() - 1U) /
                              (source.back().timestamp_sec - source.front().timestamp_sec);
  EXPECT_TRUE(std::abs(output->source_fps - expected_fps) < 1e-12);
  double expected_gap = output->end_sec - source.back().timestamp_sec;
  for (std::size_t index = 1; index < source.size(); ++index) {
    expected_gap = std::max(expected_gap, source[index].timestamp_sec -
                                          source[index - 1U].timestamp_sec);
  }
  EXPECT_TRUE(std::abs(output->max_frame_gap_sec - expected_gap) < 1e-12);
  EXPECT_EQ(output->tensor.size(), 180U * 72U * 72U * 3U);

  std::set<int> sampled_ids;
  for (const std::size_t sample : {0U, 37U, 91U, 179U}) {
    const double target = start + 6.0 * static_cast<double>(sample) / 180.0;
    const int selected = nearest_id(source, target);
    sampled_ids.insert(selected);
    expect_rgb_at(*output, sample, selected, static_cast<int>(sample % 72U),
                  static_cast<int>((sample * 7U) % 72U));
  }
  EXPECT_TRUE(sampled_ids.size() > 1U);
  for (float value : output->tensor) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

void uses_the_prestart_frame_for_the_first_target_and_has_a_coverage_policy() {
  rppg_qnn::DeepWindowBuilder builder(6.0, 180, {72, 72});
  (void)builder.add_roi(patterned_packet(0.0, 1));
  constexpr double kEnd = 6.001;
  for (int index = 0; index < 180; ++index) {
    const double timestamp = 0.034 + (kEnd - 0.034) *
                                        static_cast<double>(index) / 179.0;
    const auto output = builder.add_roi(patterned_packet(timestamp, index + 2));
    if (index == 179) {
      EXPECT_TRUE(output.has_value());
      if (output.has_value()) {
        EXPECT_TRUE(std::abs(output->start_sec - 0.001) < 1e-12);
        expect_rgb_at(*output, 0U, 1, 0, 0);
      }
    } else {
      EXPECT_TRUE(!output.has_value());
    }
  }

  rppg_qnn::DeepWindowBuilder no_prestart(6.0, 180, {72, 72});
  for (int index = 0; index < 180; ++index) {
    const double timestamp = 0.034 + (kEnd - 0.034) *
                                        static_cast<double>(index) / 179.0;
    EXPECT_TRUE(!no_prestart.add_roi(patterned_packet(timestamp, index)).has_value());
  }
  EXPECT_EQ(no_prestart.status(), std::string("start_coverage_missing"));
}

void reports_capture_quality_and_rejects_invalid_rois() {
  rppg_qnn::DeepWindowBuilder sparse(6.0, 180, {72, 72});
  for (int index = 0; index <= 6; ++index) {
    (void)sparse.add_roi(flat_packet(static_cast<double>(index)));
  }
  EXPECT_EQ(sparse.status(), std::string("low_source_fps"));

  rppg_qnn::DeepWindowBuilder gapped(6.0, 180, {72, 72});
  double gap_timestamp = 0.0;
  bool skipped_second = false;
  while (gap_timestamp <= 7.0) {
    (void)gapped.add_roi(flat_packet(gap_timestamp));
    gap_timestamp += 1.0 / 30.0;
    if (!skipped_second && gap_timestamp >= 2.0) {
      gap_timestamp += 1.0;
      skipped_second = true;
    }
  }
  EXPECT_EQ(gapped.status(), std::string("capture_gap"));

  rppg_qnn::DeepWindowBuilder invalid(6.0, 180, {72, 72});
  EXPECT_TRUE(!invalid.add_roi(flat_packet(1.0)).has_value());
  EXPECT_TRUE(!invalid.add_roi(flat_packet(0.9)).has_value());
  EXPECT_EQ(invalid.status(), std::string("nonmonotonic_timestamp"));
  rppg_qnn::RoiPacket empty{0, 2.0, {}, std::nullopt, false};
  EXPECT_TRUE(!invalid.add_roi(empty).has_value());
  EXPECT_EQ(invalid.status(), std::string("empty_roi"));

  cv::Mat float_roi(72, 72, CV_32FC3, cv::Scalar(1.0, 2.0, 3.0));
  rppg_qnn::RoiPacket malformed{0, 3.0, float_roi, std::nullopt, false};
  EXPECT_TRUE(!invalid.add_roi(malformed).has_value());
  EXPECT_EQ(invalid.status(), std::string("invalid_roi_format"));

  bool rejected_oversized_output = false;
  try {
    (void)rppg_qnn::DeepWindowBuilder(6.0, 180, {5000, 72});
  } catch (const std::invalid_argument&) {
    rejected_oversized_output = true;
  }
  EXPECT_TRUE(rejected_oversized_output);
}

void recovers_from_error_and_uses_earlier_frames_for_ties() {
  rppg_qnn::DeepWindowBuilder recovering(6.0, 180, {72, 72});
  rppg_qnn::RoiPacket empty{0, 0.0, {}, std::nullopt, false};
  EXPECT_TRUE(!recovering.add_roi(empty).has_value());
  EXPECT_EQ(recovering.status(), std::string("empty_roi"));
  std::optional<rppg_qnn::DeepInput> recovered;
  for (int index = 0; index <= 180; ++index) {
    recovered = recovering.add_roi(patterned_packet(index / 30.0, index));
  }
  EXPECT_TRUE(recovered.has_value());
  EXPECT_EQ(recovering.status(), std::string("ready"));

  rppg_qnn::DeepWindowBuilder tie_builder(6.0, 180, {72, 72});
  EXPECT_TRUE(!tie_builder.add_roi(patterned_packet(0.0, 1)).has_value());
  EXPECT_TRUE(!tie_builder.add_roi(patterned_packet(0.020, 2)).has_value());
  EXPECT_TRUE(!tie_builder.add_roi(patterned_packet(0.04666666666666667, 3)).has_value());
  std::optional<rppg_qnn::DeepInput> tied;
  for (int index = 0; index < 178; ++index) {
    const double timestamp = 0.080 + (6.0 - 0.080) * index / 177.0;
    tied = tie_builder.add_roi(patterned_packet(timestamp, index + 4));
  }
  EXPECT_TRUE(tied.has_value());
  if (tied.has_value()) {
    expect_rgb_at(*tied, 1U, 2, 0, 0);
  }
}

void owns_roi_pixels_after_the_caller_reuses_the_buffer() {
  rppg_qnn::DeepWindowBuilder builder(6.0, 180, {72, 72});
  cv::Mat roi(72, 72, CV_8UC3, cv::Scalar(9, 8, 7));
  std::optional<rppg_qnn::DeepInput> output;
  for (int index = 0; index <= 180; ++index) {
    output = builder.add_roi({0, index / 30.0, roi, std::nullopt, false});
    roi.setTo(cv::Scalar(1, 2, 3));
    if (index != 180) {
      roi.setTo(cv::Scalar(9, 8, 7));
    }
  }
  EXPECT_TRUE(output.has_value());
  if (output.has_value()) {
    EXPECT_EQ(output->tensor[0], 7.0F);
    EXPECT_EQ(output->tensor[1], 8.0F);
    EXPECT_EQ(output->tensor[2], 9.0F);
  }
}

}  // namespace

int main() {
  builds_first_ready_rgb_nhwc_window_from_jittered_capture();
  uses_the_prestart_frame_for_the_first_target_and_has_a_coverage_policy();
  reports_capture_quality_and_rejects_invalid_rois();
  recovers_from_error_and_uses_earlier_frames_for_ties();
  owns_roi_pixels_after_the_caller_reuses_the_buffer();
  return test_support::finish();
}
