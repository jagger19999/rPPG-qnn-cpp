#pragma once

#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/frame_source.hpp"

#include <filesystem>
#include <functional>
#include <optional>
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

  RoiPacket process(const FramePacket& frame) override;

 private:
  FaceDetector detector_;
  std::optional<FaceBox> last_face_;
  std::optional<cv::Size> last_frame_size_;
  int fallback_frames_remaining_{0};
  bool force_detection_{true};
};

}  // namespace rppg_qnn
