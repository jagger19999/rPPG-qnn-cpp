#include "rppg_qnn/roi_processor.hpp"

#include "rppg_qnn/error.hpp"

#include <algorithm>
#include <cstdint>
#include <system_error>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/imgproc.hpp>

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

std::optional<FaceBox> largest_usable_face(const std::vector<cv::Rect>& faces,
                                           const cv::Size frame_size) {
  std::optional<FaceBox> selected;
  std::int64_t selected_area = 0;
  for (const cv::Rect& candidate : faces) {
    const FaceBox face{candidate.x, candidate.y, candidate.width, candidate.height, 1.0};
    if (!cheek_roi(face, frame_size).has_value()) {
      continue;
    }
    const std::int64_t area = static_cast<std::int64_t>(candidate.width) *
                              static_cast<std::int64_t>(candidate.height);
    if (!selected.has_value() || area > selected_area) {
      selected = face;
      selected_area = area;
    }
  }
  return selected;
}

RoiPacket empty_packet(const FramePacket& frame) {
  return RoiPacket{frame.frame_id, frame.timestamp_sec, {}, std::nullopt, false};
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

RoiProcessor::RoiProcessor(const std::filesystem::path& cascade_path) {
  std::error_code error;
  if (!std::filesystem::is_regular_file(cascade_path, error)) {
    throw AppError(ErrorCode::ConfigInvalid,
                   "Unable to load Haar cascade: " + cascade_path.string());
  }
  try {
    if (!cascade_.load(cascade_path.string())) {
      throw AppError(ErrorCode::ConfigInvalid,
                     "Unable to load Haar cascade: " + cascade_path.string());
    }
  } catch (const cv::Exception&) {
    throw AppError(ErrorCode::ConfigInvalid,
                   "Unable to load Haar cascade: " + cascade_path.string());
  }
}

RoiPacket RoiProcessor::process(const FramePacket& frame) {
  if (frame.bgr.empty()) {
    return empty_packet(frame);
  }

  const bool should_detect = (processed_frames_ % 10U) == 0U;
  ++processed_frames_;
  if (should_detect) {
    cv::Mat gray;
    cv::cvtColor(frame.bgr, gray, cv::COLOR_BGR2GRAY);
    std::vector<cv::Rect> detected_faces;
    cascade_.detectMultiScale(gray, detected_faces, 1.1, 3, 0, cv::Size(30, 30));
    last_face_ = largest_usable_face(detected_faces, frame.bgr.size());
    if (!last_face_.has_value()) {
      return empty_packet(frame);
    }
  } else if (!last_face_.has_value()) {
    return empty_packet(frame);
  }

  const auto roi = cheek_roi(*last_face_, frame.bgr.size());
  if (!roi.has_value()) {
    last_face_.reset();
    return empty_packet(frame);
  }

  return RoiPacket{frame.frame_id,
                   frame.timestamp_sec,
                   frame.bgr(*roi).clone(),
                   last_face_,
                   !should_detect};
}

}  // namespace rppg_qnn
