#include "rppg_qnn/deep_window_builder.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <opencv2/imgproc.hpp>

namespace rppg_qnn {
namespace {

constexpr double kMinimumSourceFps = 15.0;
constexpr double kMaximumGapSeconds = 0.75;
constexpr int kMaximumOutputDimension = 4096;
constexpr std::size_t kMaximumTensorValues = 100000000U;

}  // namespace

DeepWindowBuilder::DeepWindowBuilder(double window_sec, std::size_t sample_count,
                                     cv::Size output_size)
    : window_sec_(window_sec), sample_count_(sample_count), output_size_(output_size) {
  if (!std::isfinite(window_sec_) || window_sec_ <= 0.0 || sample_count_ == 0U ||
      output_size_.width <= 0 || output_size_.height <= 0 ||
      output_size_.width > kMaximumOutputDimension ||
      output_size_.height > kMaximumOutputDimension) {
    throw std::invalid_argument("Deep window parameters must be positive");
  }
  const std::size_t values_per_frame =
      static_cast<std::size_t>(output_size_.width) *
      static_cast<std::size_t>(output_size_.height) * 3U;
  if (values_per_frame > kMaximumTensorValues / sample_count_) {
    throw std::invalid_argument("Deep window tensor is too large");
  }
}

std::optional<DeepInput> DeepWindowBuilder::add_roi(const RoiPacket& packet) {
  if (!std::isfinite(packet.timestamp_sec)) {
    status_ = "invalid_timestamp";
    return std::nullopt;
  }
  if (packet.roi_bgr.empty()) {
    status_ = "empty_roi";
    return std::nullopt;
  }
  if (packet.roi_bgr.type() != CV_8UC3) {
    status_ = "invalid_roi_format";
    return std::nullopt;
  }
  if (!frames_.empty() && packet.timestamp_sec <= frames_.back().timestamp_sec) {
    status_ = "nonmonotonic_timestamp";
    return std::nullopt;
  }
  frames_.push_back({packet.timestamp_sec, packet.roi_bgr.clone()});
  while (!frames_.empty() &&
         frames_.front().timestamp_sec < packet.timestamp_sec - 2.0 * window_sec_) {
    frames_.pop_front();
  }
  if (packet.timestamp_sec - frames_.front().timestamp_sec < window_sec_) {
    status_ = "sampling";
    return std::nullopt;
  }

  const double end = packet.timestamp_sec;
  const double start = end - window_sec_;
  std::vector<const Frame*> source;
  source.reserve(frames_.size());
  for (const Frame& frame : frames_) {
    if (frame.timestamp_sec >= start && frame.timestamp_sec <= end) {
      source.push_back(&frame);
    }
  }
  if (source.size() < 2U) {
    status_ = "sampling";
    return std::nullopt;
  }

  double max_gap = std::max(source.front()->timestamp_sec - start,
                            end - source.back()->timestamp_sec);
  for (std::size_t index = 1; index < source.size(); ++index) {
    max_gap = std::max(max_gap, source[index]->timestamp_sec -
                                     source[index - 1U]->timestamp_sec);
  }
  const double source_span = source.back()->timestamp_sec - source.front()->timestamp_sec;
  const double source_fps = source_span > 0.0
                                ? static_cast<double>(source.size() - 1U) / source_span
                                : 0.0;
  if (source_fps < kMinimumSourceFps) {
    status_ = "low_source_fps";
    return std::nullopt;
  }
  if (max_gap > kMaximumGapSeconds) {
    status_ = "capture_gap";
    return std::nullopt;
  }

  DeepInput input;
  input.start_sec = start;
  input.end_sec = end;
  input.source_fps = source_fps;
  input.max_frame_gap_sec = max_gap;
  input.source_frame_count = source.size();
  input.shape = {static_cast<std::int64_t>(sample_count_), output_size_.height,
                 output_size_.width, 3};
  const std::size_t values_per_frame =
      static_cast<std::size_t>(output_size_.width) *
      static_cast<std::size_t>(output_size_.height) * 3U;
  input.tensor.reserve(sample_count_ * values_per_frame);
  for (std::size_t index = 0; index < sample_count_; ++index) {
    const double target = start + window_sec_ * static_cast<double>(index) /
                                      static_cast<double>(sample_count_);
    const auto closest = std::min_element(
        source.begin(), source.end(), [target](const Frame* left, const Frame* right) {
          return std::abs(left->timestamp_sec - target) <
                 std::abs(right->timestamp_sec - target);
        });
    cv::Mat resized;
    cv::resize((*closest)->bgr, resized, output_size_, 0.0, 0.0, cv::INTER_LINEAR);
    cv::Mat rgb;
    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
    for (int row = 0; row < rgb.rows; ++row) {
      const cv::Vec3b* pixels = rgb.ptr<cv::Vec3b>(row);
      for (int column = 0; column < rgb.cols; ++column) {
        input.tensor.push_back(static_cast<float>(pixels[column][0]));
        input.tensor.push_back(static_cast<float>(pixels[column][1]));
        input.tensor.push_back(static_cast<float>(pixels[column][2]));
      }
    }
  }
  status_ = "ready";
  return input;
}

const std::string& DeepWindowBuilder::status() const { return status_; }

}  // namespace rppg_qnn
