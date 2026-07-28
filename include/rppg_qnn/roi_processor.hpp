#pragma once

#include "rppg_qnn/frame_source.hpp"

#include <filesystem>
#include <functional>
#include <optional>
#include <utility>
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
  RoiPacket() = default;
  RoiPacket(std::uint64_t requested_frame_id, double requested_timestamp_sec,
            cv::Mat requested_roi_bgr, std::optional<FaceBox> requested_face,
            bool requested_used_fallback, int requested_face_count = 0,
            double requested_motion_px = 0.0,
            cv::Mat requested_deep_roi_bgr = {})
      : frame_id(requested_frame_id),
        timestamp_sec(requested_timestamp_sec),
        roi_bgr(std::move(requested_roi_bgr)),
        face(std::move(requested_face)),
        used_fallback(requested_used_fallback),
        face_count(requested_face_count),
        motion_px(requested_motion_px),
        deep_roi_bgr(std::move(requested_deep_roi_bgr)) {}
  std::uint64_t frame_id{0};
  double timestamp_sec{0.0};
  cv::Mat roi_bgr;
  std::optional<FaceBox> face;
  bool used_fallback{false};
  int face_count{0};
  double motion_px{0.0};
  cv::Mat deep_roi_bgr;
};

using FaceDetector = std::function<std::vector<FaceBox>(const cv::Mat&)>;

std::optional<cv::Rect> cheek_roi(const FaceBox& face, cv::Size frame_size);

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
