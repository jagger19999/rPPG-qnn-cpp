#pragma once

#include "rppg_qnn/frame_source.hpp"

#include <filesystem>
#include <optional>

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

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

std::optional<cv::Rect> cheek_roi(const FaceBox& face, cv::Size frame_size);

class RoiProcessor {
 public:
  explicit RoiProcessor(const std::filesystem::path& cascade_path);

  RoiPacket process(const FramePacket& frame);

 private:
  cv::CascadeClassifier cascade_;
  std::optional<FaceBox> last_face_;
  std::uint64_t processed_frames_{0};
};

}  // namespace rppg_qnn
