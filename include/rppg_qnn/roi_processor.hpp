#pragma once

#include "rppg_qnn/frame_source.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <vector>

#include <opencv2/core.hpp>

namespace rppg_qnn {

struct FaceBox {
  int x{0};
  int y{0};
  int width{0};
  int height{0};
  double confidence{0.0};
};

struct RoiPacket {
  std::uint64_t frame_id{0};
  double timestamp_sec{0.0};
  cv::Mat roi_bgr;
  std::optional<FaceBox> face;
  bool used_fallback{false};
};

using FaceDetector = std::function<std::vector<FaceBox>(const cv::Mat&)>;

std::optional<cv::Rect> cheek_roi(const FaceBox& face, cv::Size frame_size);

class RoiProcessor {
 public:
  explicit RoiProcessor(const std::filesystem::path& cascade_path);
  explicit RoiProcessor(FaceDetector detector);

  RoiPacket process(const FramePacket& frame);

 private:
  FaceDetector detector_;
  std::optional<FaceBox> last_face_;
  std::optional<cv::Size> last_frame_size_;
  int fallback_frames_remaining_{0};
  bool force_detection_{true};
};

}  // namespace rppg_qnn
