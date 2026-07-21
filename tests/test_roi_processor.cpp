#include "rppg_qnn/error.hpp"
#include "rppg_qnn/roi_processor.hpp"

#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "test_support.hpp"

namespace {

using rppg_qnn::AppError;
using rppg_qnn::ErrorCode;
using rppg_qnn::FaceBox;
using rppg_qnn::FaceDetector;
using rppg_qnn::FramePacket;
using rppg_qnn::RoiProcessor;
using rppg_qnn::cheek_roi;

void returns_lower_central_cheek_region() {
  const FaceBox face{50, 40, 100, 120, 0.9};
  const auto roi = cheek_roi(face, cv::Size(200, 200));

  EXPECT_TRUE(roi.has_value());
  if (!roi.has_value()) {
    return;
  }

  EXPECT_TRUE(roi->x >= 0 && roi->y >= 0);
  EXPECT_TRUE(roi->x + roi->width <= 200 && roi->y + roi->height <= 200);
  EXPECT_TRUE(roi->x >= face.x);
  EXPECT_TRUE(roi->x + roi->width <= face.x + face.width);
  EXPECT_EQ(roi->x + roi->width / 2, face.x + face.width / 2);
  EXPECT_TRUE(roi->y >= face.y + face.height / 2);
  EXPECT_TRUE(roi->y + roi->height <= face.y + face.height);
  EXPECT_TRUE(roi->area() >= (face.width * face.height * 5) / 100);
}

void clips_partially_out_of_bounds_faces() {
  const auto roi = cheek_roi(FaceBox{-20, 30, 100, 100, 0.0}, cv::Size(80, 100));

  EXPECT_TRUE(roi.has_value());
  if (!roi.has_value()) {
    return;
  }
  EXPECT_TRUE(roi->x >= 0 && roi->y >= 0);
  EXPECT_TRUE(roi->x + roi->width <= 80 && roi->y + roi->height <= 100);
  EXPECT_TRUE(roi->area() > 0);
}

void rejects_empty_invalid_and_overflowing_geometry() {
  EXPECT_TRUE(!cheek_roi(FaceBox{0, 0, 0, 10, 0.0}, cv::Size(200, 200)).has_value());
  EXPECT_TRUE(!cheek_roi(FaceBox{0, 0, 10, -1, 0.0}, cv::Size(200, 200)).has_value());
  EXPECT_TRUE(!cheek_roi(FaceBox{0, 0, 10, 10, 0.0}, cv::Size()).has_value());
  EXPECT_TRUE(!cheek_roi(FaceBox{std::numeric_limits<int>::max() - 2,
                                 0,
                                 20,
                                 20,
                                 0.0},
                         cv::Size(200, 200))
                   .has_value());
}

void rejects_unloadable_cascade_with_descriptive_config_error() {
  const std::filesystem::path path = "/definitely/missing/rppg_cascade.xml";
  try {
    rppg_qnn::RoiProcessor processor(path);
    (void)processor;
    EXPECT_TRUE(false);
  } catch (const AppError& error) {
    EXPECT_EQ(error.code(), ErrorCode::ConfigInvalid);
    EXPECT_TRUE(std::string(error.what()).find(path.string()) != std::string::npos);
  }
}

void reuses_clipped_largest_face_until_the_next_detection_interval() {
  int detector_calls = 0;
  FaceDetector detector = [&detector_calls](const cv::Mat&) {
    ++detector_calls;
    if (detector_calls == 1) {
      return std::vector<FaceBox>{
          FaceBox{50, 40, 90, 110, 0.4},
          FaceBox{-20, 40, 120, 120, 0.8},
          FaceBox{0, 0, 0, 100, 0.0},
      };
    }
    return std::vector<FaceBox>{};
  };
  RoiProcessor processor(detector);

  FramePacket first{7,
                    1.25,
                    cv::Mat(200, 200, CV_8UC3, cv::Scalar(10, 20, 30))};
  const auto detected = processor.process(first);
  EXPECT_EQ(detector_calls, 1);
  EXPECT_EQ(detected.frame_id, first.frame_id);
  EXPECT_EQ(detected.timestamp_sec, first.timestamp_sec);
  EXPECT_TRUE(detected.face.has_value());
  EXPECT_TRUE(!detected.used_fallback);
  EXPECT_TRUE(!detected.roi_bgr.empty());
  if (detected.face.has_value()) {
    EXPECT_EQ(detected.face->x, 0);
    EXPECT_EQ(detected.face->y, 40);
    EXPECT_EQ(detected.face->width, 100);
    EXPECT_EQ(detected.face->height, 120);
  }

  first.bgr.setTo(cv::Scalar());
  EXPECT_TRUE(detected.roi_bgr.at<cv::Vec3b>(0, 0) == cv::Vec3b(10, 20, 30));

  for (int index = 0; index < 9; ++index) {
    FramePacket reuse{static_cast<std::uint64_t>(index + 8),
                      2.0 + index,
                      cv::Mat(200, 200, CV_8UC3, cv::Scalar(1, 2, 3))};
    const auto fallback = processor.process(reuse);
    EXPECT_EQ(detector_calls, 1);
    EXPECT_EQ(fallback.frame_id, reuse.frame_id);
    EXPECT_EQ(fallback.timestamp_sec, reuse.timestamp_sec);
    EXPECT_TRUE(fallback.face.has_value());
    EXPECT_TRUE(fallback.used_fallback);
    EXPECT_TRUE(!fallback.roi_bgr.empty());
  }

  FramePacket eleventh{17, 12.0, cv::Mat(200, 200, CV_8UC3, cv::Scalar())};
  const auto no_face = processor.process(eleventh);
  EXPECT_EQ(detector_calls, 2);
  EXPECT_TRUE(!no_face.face.has_value());
  EXPECT_TRUE(!no_face.used_fallback);
  EXPECT_TRUE(no_face.roi_bgr.empty());

  const auto cleared = processor.process(eleventh);
  EXPECT_EQ(detector_calls, 2);
  EXPECT_TRUE(!cleared.face.has_value());
  EXPECT_TRUE(cleared.roi_bgr.empty());
}

void empty_input_and_empty_detection_never_return_a_full_frame() {
  int detector_calls = 0;
  FaceDetector detector = [&detector_calls](const cv::Mat&) {
    ++detector_calls;
    return std::vector<FaceBox>{};
  };
  RoiProcessor processor(detector);

  const FramePacket empty{9, 3.5, {}};
  const auto empty_result = processor.process(empty);
  EXPECT_EQ(detector_calls, 0);
  EXPECT_EQ(empty_result.frame_id, empty.frame_id);
  EXPECT_EQ(empty_result.timestamp_sec, empty.timestamp_sec);
  EXPECT_TRUE(empty_result.roi_bgr.empty());
  EXPECT_TRUE(!empty_result.face.has_value());
  EXPECT_TRUE(!empty_result.used_fallback);

  const FramePacket nonempty{10, 4.0, cv::Mat(200, 200, CV_8UC3, cv::Scalar(9, 8, 7))};
  const auto no_face = processor.process(nonempty);
  EXPECT_EQ(detector_calls, 1);
  EXPECT_TRUE(no_face.roi_bgr.empty());
  EXPECT_TRUE(!no_face.face.has_value());
  EXPECT_TRUE(!no_face.used_fallback);
}

void rejects_an_empty_injected_detector() {
  EXPECT_APP_ERROR(RoiProcessor(FaceDetector{}), ErrorCode::ConfigInvalid);
}

}  // namespace

int main() {
  returns_lower_central_cheek_region();
  clips_partially_out_of_bounds_faces();
  rejects_empty_invalid_and_overflowing_geometry();
  rejects_unloadable_cascade_with_descriptive_config_error();
  reuses_clipped_largest_face_until_the_next_detection_interval();
  empty_input_and_empty_detection_never_return_a_full_frame();
  rejects_an_empty_injected_detector();
  return test_support::finish();
}
