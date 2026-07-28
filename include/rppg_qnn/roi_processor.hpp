#pragma once

#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/frame_source.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace rppg_qnn {

using FaceDetector = std::function<std::vector<FaceBox>(const cv::Mat&)>;

std::optional<cv::Rect> cheek_roi(const FaceBox& face, cv::Size frame_size);
std::optional<cv::Rect> expanded_face_roi(const FaceBox& face,
                                          cv::Size frame_size,
                                          double scale = 1.5);

class IRoiProcessor {
 public:
  virtual ~IRoiProcessor() = default;
  virtual RoiPacket process(const FramePacket& frame) = 0;
};

class RoiProcessor final : public IRoiProcessor {
 public:
  explicit RoiProcessor(const std::filesystem::path& cascade_path);
  explicit RoiProcessor(FaceDetector detector);
  RoiProcessor(FaceDetector detector, bool enable_tracking);

  RoiPacket process(const FramePacket& frame) override;

 private:
  FaceDetector detector_;
  std::optional<FaceBox> last_face_;
  std::optional<cv::Size> last_frame_size_;
  int fallback_frames_remaining_{0};
  bool force_detection_{true};
  bool tracking_enabled_{false};
  int last_face_count_{0};
  cv::Mat previous_gray_;
  std::vector<cv::Point2f> tracking_points_;
};

}  // namespace rppg_qnn
