#include "rppg_qnn/roi_processor.hpp"

#include "rppg_qnn/error.hpp"

#include <algorithm>
#include <cstdint>
#include <cmath>
#include <memory>
#include <system_error>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <opencv2/video/tracking.hpp>

namespace rppg_qnn {
namespace {

std::optional<cv::Rect> clipped_face_rect(const FaceBox& face,
                                          const cv::Size frame_size) {
  if (frame_size.width <= 0 || frame_size.height <= 0 || face.width <= 0 ||
      face.height <= 0) {
    return std::nullopt;
  }

  const std::int64_t left = static_cast<std::int64_t>(face.x);
  const std::int64_t top = static_cast<std::int64_t>(face.y);
  const std::int64_t right = left + static_cast<std::int64_t>(face.width);
  const std::int64_t bottom = top + static_cast<std::int64_t>(face.height);
  const std::int64_t clipped_left = std::max<std::int64_t>(0, left);
  const std::int64_t clipped_top = std::max<std::int64_t>(0, top);
  const std::int64_t clipped_right = std::min<std::int64_t>(frame_size.width, right);
  const std::int64_t clipped_bottom = std::min<std::int64_t>(frame_size.height, bottom);

  if (clipped_right <= clipped_left || clipped_bottom <= clipped_top) {
    return std::nullopt;
  }

  return cv::Rect(static_cast<int>(clipped_left),
                  static_cast<int>(clipped_top),
                  static_cast<int>(clipped_right - clipped_left),
                  static_cast<int>(clipped_bottom - clipped_top));
}

std::optional<FaceBox> clipped_face_box(const FaceBox& face,
                                        const cv::Size frame_size) {
  const auto clipped = clipped_face_rect(face, frame_size);
  if (!clipped.has_value()) {
    return std::nullopt;
  }
  return FaceBox{clipped->x, clipped->y, clipped->width, clipped->height,
                 face.confidence};
}

std::optional<FaceBox> largest_usable_face(const std::vector<FaceBox>& faces,
                                           const cv::Size frame_size) {
  std::optional<FaceBox> selected;
  std::int64_t selected_area = 0;
  for (const FaceBox& candidate : faces) {
    const auto clipped = clipped_face_box(candidate, frame_size);
    if (!clipped.has_value() || !cheek_roi(*clipped, frame_size).has_value()) {
      continue;
    }
    const std::int64_t area = static_cast<std::int64_t>(clipped->width) *
                              static_cast<std::int64_t>(clipped->height);
    if (!selected.has_value() || area > selected_area) {
      selected = *clipped;
      selected_area = area;
    }
  }
  return selected;
}

RoiPacket empty_packet(const FramePacket& frame) {
  return RoiPacket{frame.frame_id, frame.timestamp_sec, {}, std::nullopt, false,
                   0, 0.0, {}};
}

FaceDetector make_haar_detector(const std::filesystem::path& cascade_path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(cascade_path, error)) {
    throw AppError(ErrorCode::ConfigInvalid,
                   "Unable to load Haar cascade: " + cascade_path.string());
  }

  auto cascade = std::make_shared<cv::CascadeClassifier>();
  try {
    if (!cascade->load(cascade_path.string())) {
      throw AppError(ErrorCode::ConfigInvalid,
                     "Unable to load Haar cascade: " + cascade_path.string());
    }
  } catch (const cv::Exception&) {
    throw AppError(ErrorCode::ConfigInvalid,
                   "Unable to load Haar cascade: " + cascade_path.string());
  }

  return [cascade](const cv::Mat& bgr) {
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> rectangles;
    cascade->detectMultiScale(gray, rectangles, 1.1, 3, 0, cv::Size(30, 30));

    std::vector<FaceBox> faces;
    faces.reserve(rectangles.size());
    for (const cv::Rect& rectangle : rectangles) {
      faces.push_back(
          FaceBox{rectangle.x, rectangle.y, rectangle.width, rectangle.height, 1.0});
    }
    return faces;
  };
}

}  // namespace

std::optional<cv::Rect> cheek_roi(const FaceBox& face, cv::Size frame_size) {
  const auto clipped_face = clipped_face_rect(face, frame_size);
  if (!clipped_face.has_value()) {
    return std::nullopt;
  }

  const std::int64_t face_width = clipped_face->width;
  const std::int64_t face_height = clipped_face->height;
  const std::int64_t roi_width = std::max<std::int64_t>(1, (face_width * 3) / 5);
  const std::int64_t roi_height = std::max<std::int64_t>(1, (face_height * 2) / 5);
  const std::int64_t roi_x = clipped_face->x + (face_width - roi_width) / 2;
  const std::int64_t roi_y = clipped_face->y + (face_height * 11) / 20;

  return cv::Rect(static_cast<int>(roi_x),
                  static_cast<int>(roi_y),
                  static_cast<int>(roi_width),
                  static_cast<int>(roi_height));
}

RoiProcessor::RoiProcessor(const std::filesystem::path& cascade_path)
    : detector_(make_haar_detector(cascade_path)), tracking_enabled_(true) {}

RoiProcessor::RoiProcessor(FaceDetector detector)
    : RoiProcessor(std::move(detector), false) {}

RoiProcessor::RoiProcessor(FaceDetector detector, bool enable_tracking)
    : detector_(std::move(detector)), tracking_enabled_(enable_tracking) {
  if (!detector_) {
    throw AppError(ErrorCode::ConfigInvalid, "Face detector must not be empty");
  }
}

RoiPacket RoiProcessor::process(const FramePacket& frame) {
  if (frame.bgr.empty()) {
    last_face_.reset();
    last_frame_size_.reset();
    fallback_frames_remaining_ = 0;
    force_detection_ = true;
    previous_gray_.release();
    tracking_points_.clear();
    last_face_count_ = 0;
    return empty_packet(frame);
  }

  const cv::Size frame_size = frame.bgr.size();
  const bool size_changed =
      last_frame_size_.has_value() &&
      (last_frame_size_->width != frame_size.width ||
       last_frame_size_->height != frame_size.height);
  if (size_changed) {
    last_face_.reset();
    fallback_frames_remaining_ = 0;
    force_detection_ = true;
    previous_gray_.release();
    tracking_points_.clear();
    last_face_count_ = 0;
  }

  cv::Mat current_gray;
  double motion_px = 0.0;
  bool tracked = false;
  if (tracking_enabled_) {
    cv::cvtColor(frame.bgr, current_gray, cv::COLOR_BGR2GRAY);
    if (!force_detection_ && fallback_frames_remaining_ > 0 &&
        last_face_.has_value() && !previous_gray_.empty() &&
        tracking_points_.size() >= 6U) {
      std::vector<cv::Point2f> next_points;
      std::vector<unsigned char> statuses;
      std::vector<float> errors;
      cv::calcOpticalFlowPyrLK(previous_gray_, current_gray, tracking_points_,
                               next_points, statuses, errors);
      cv::Point2d displacement;
      std::vector<cv::Point2f> valid_points;
      for (std::size_t index = 0; index < statuses.size(); ++index) {
        if (statuses[index] != 0U && std::isfinite(errors[index]) &&
            errors[index] < 30.0F) {
          displacement += cv::Point2d(next_points[index] - tracking_points_[index]);
          valid_points.push_back(next_points[index]);
        }
      }
      if (valid_points.size() >= 6U) {
        displacement *= 1.0 / static_cast<double>(valid_points.size());
        FaceBox moved = *last_face_;
        moved.x += static_cast<int>(std::lround(displacement.x));
        moved.y += static_cast<int>(std::lround(displacement.y));
        last_face_ = clipped_face_box(moved, frame_size);
        if (last_face_.has_value()) {
          motion_px = cv::norm(displacement);
          tracking_points_ = std::move(valid_points);
          previous_gray_ = current_gray.clone();
          tracked = true;
        }
      }
      if (!tracked) {
        fallback_frames_remaining_ = 0;
        tracking_points_.clear();
      }
    }
  }

  const bool should_detect = force_detection_ || fallback_frames_remaining_ == 0;
  if (should_detect) {
    last_face_.reset();
    fallback_frames_remaining_ = 0;
    force_detection_ = true;
    const auto detected_faces = detector_(frame.bgr);
    last_face_count_ = static_cast<int>(detected_faces.size());
    last_face_ = largest_usable_face(detected_faces, frame_size);
    last_frame_size_ = frame_size;
    fallback_frames_remaining_ = tracking_enabled_ ? 29 : 9;
    force_detection_ = false;
    if (!last_face_.has_value()) {
      previous_gray_ = current_gray.clone();
      return empty_packet(frame);
    }
    if (tracking_enabled_) {
      cv::Mat mask(current_gray.size(), CV_8UC1, cv::Scalar());
      const auto face_rect = clipped_face_rect(*last_face_, frame_size);
      if (face_rect.has_value()) {
        mask(*face_rect).setTo(cv::Scalar(255));
        cv::goodFeaturesToTrack(current_gray, tracking_points_, 80, 0.01, 5.0,
                                mask);
      }
      previous_gray_ = current_gray.clone();
    }
  }

  if (!should_detect) {
    --fallback_frames_remaining_;
  }
  if (!last_face_.has_value()) {
    return empty_packet(frame);
  }
  const auto roi = cheek_roi(*last_face_, frame.bgr.size());
  const auto deep_roi = clipped_face_rect(*last_face_, frame.bgr.size());
  if (!roi.has_value() || !deep_roi.has_value()) {
    last_face_.reset();
    fallback_frames_remaining_ = 0;
    force_detection_ = true;
    return empty_packet(frame);
  }

  return RoiPacket{frame.frame_id,
                   frame.timestamp_sec,
                   frame.bgr(*roi).clone(),
                   last_face_,
                   !should_detect,
                   std::max(last_face_count_, 1),
                   motion_px,
                   frame.bgr(*deep_roi).clone()};
}

}  // namespace rppg_qnn
