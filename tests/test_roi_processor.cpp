#include "rppg_qnn/error.hpp"
#include "rppg_qnn/roi_processor.hpp"

#include <filesystem>
#include <limits>
#include <stdexcept>
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
using rppg_qnn::expanded_face_roi;

void returns_python_compatible_expanded_face_geometry() {
  EXPECT_EQ(expanded_face_roi(FaceBox{100, 80, 40, 60, 0.0},
                              cv::Size(300, 300)),
            std::optional<cv::Rect>(cv::Rect(90, 65, 60, 90)));

  EXPECT_EQ(expanded_face_roi(FaceBox{-10, 40, 20, 20, 0.0},
                              cv::Size(100, 100)),
            std::optional<cv::Rect>(cv::Rect(0, 35, 30, 30)));
  EXPECT_EQ(expanded_face_roi(FaceBox{40, -10, 20, 20, 0.0},
                              cv::Size(100, 100)),
            std::optional<cv::Rect>(cv::Rect(35, 0, 30, 30)));
  EXPECT_EQ(expanded_face_roi(FaceBox{90, 40, 20, 20, 0.0},
                              cv::Size(100, 100)),
            std::optional<cv::Rect>(cv::Rect(85, 35, 15, 30)));
  EXPECT_EQ(expanded_face_roi(FaceBox{40, 90, 20, 20, 0.0},
                              cv::Size(100, 100)),
            std::optional<cv::Rect>(cv::Rect(35, 85, 30, 15)));

  // Python round() uses ties-to-even: round(3 * 1.5) == 4 and round(9.5) == 10.
  EXPECT_EQ(expanded_face_roi(FaceBox{10, 10, 3, 3, 0.0},
                              cv::Size(30, 30)),
            std::optional<cv::Rect>(cv::Rect(10, 10, 4, 4)));
  EXPECT_EQ(expanded_face_roi(FaceBox{10, 10, 5, 5, 0.0},
                              cv::Size(30, 30)),
            std::optional<cv::Rect>(cv::Rect(8, 8, 8, 8)));
}

void rejects_invalid_expanded_face_geometry() {
  const FaceBox valid{10, 10, 20, 20, 0.0};
  EXPECT_TRUE(!expanded_face_roi(valid, cv::Size(), 1.5).has_value());
  EXPECT_TRUE(!expanded_face_roi(FaceBox{0, 0, 0, 20, 0.0},
                                 cv::Size(100, 100), 1.5).has_value());
  EXPECT_TRUE(!expanded_face_roi(valid, cv::Size(100, 100), 0.0).has_value());
  EXPECT_TRUE(!expanded_face_roi(valid, cv::Size(100, 100), -1.0).has_value());
  EXPECT_TRUE(!expanded_face_roi(valid, cv::Size(100, 100),
                                 std::numeric_limits<double>::infinity())
                   .has_value());
  EXPECT_TRUE(!expanded_face_roi(valid, cv::Size(100, 100),
                                 std::numeric_limits<double>::quiet_NaN())
                   .has_value());
  EXPECT_TRUE(!expanded_face_roi(valid, cv::Size(100, 100),
                                 std::numeric_limits<double>::max())
                   .has_value());
  EXPECT_TRUE(!expanded_face_roi(FaceBox{100, 100, 10, 10, 0.0},
                                 cv::Size(20, 20), 1.5).has_value());

  // The existing traditional geometry is an explicit regression contract.
  EXPECT_EQ(cheek_roi(FaceBox{50, 40, 100, 120, 0.9}, cv::Size(200, 200)),
            std::optional<cv::Rect>(cv::Rect(70, 106, 60, 48)));
  EXPECT_EQ(cheek_roi(FaceBox{-20, 30, 100, 100, 0.0}, cv::Size(80, 100)),
            std::optional<cv::Rect>(cv::Rect(16, 68, 48, 28)));
}

void process_returns_distinct_cloned_traditional_and_deep_crops() {
  const FaceBox face{20, 20, 20, 20, 0.9};
  RoiProcessor processor([face](const cv::Mat&) {
    return std::vector<FaceBox>{face};
  });
  cv::Mat patterned(60, 60, CV_8UC3);
  for (int row = 0; row < patterned.rows; ++row) {
    for (int column = 0; column < patterned.cols; ++column) {
      patterned.at<cv::Vec3b>(row, column) = cv::Vec3b(
          static_cast<unsigned char>(row), static_cast<unsigned char>(column),
          static_cast<unsigned char>(row + column));
    }
  }
  const cv::Mat original = patterned.clone();
  const auto packet = processor.process(FramePacket{1, 1.0, patterned});
  const auto traditional_rect = cheek_roi(face, patterned.size());
  const auto deep_rect = expanded_face_roi(face, patterned.size());

  EXPECT_TRUE(traditional_rect.has_value());
  EXPECT_TRUE(deep_rect.has_value());
  EXPECT_TRUE(!packet.roi_bgr.empty());
  EXPECT_TRUE(!packet.deep_roi_bgr.empty());
  if (!traditional_rect.has_value() || !deep_rect.has_value()) {
    return;
  }
  EXPECT_EQ(cv::norm(packet.roi_bgr, original(*traditional_rect), cv::NORM_INF), 0.0);
  EXPECT_EQ(cv::norm(packet.deep_roi_bgr, original(*deep_rect), cv::NORM_INF), 0.0);
  EXPECT_TRUE(packet.roi_bgr.data != patterned.data);
  EXPECT_TRUE(packet.deep_roi_bgr.data != patterned.data);
  EXPECT_TRUE(packet.roi_bgr.data != packet.deep_roi_bgr.data);

  patterned.setTo(cv::Scalar(255, 255, 255));
  EXPECT_EQ(cv::norm(packet.roi_bgr, original(*traditional_rect), cv::NORM_INF), 0.0);
  EXPECT_EQ(cv::norm(packet.deep_roi_bgr, original(*deep_rect), cv::NORM_INF), 0.0);
}

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
  EXPECT_TRUE(no_face.deep_roi_bgr.empty());

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
  EXPECT_TRUE(empty_result.deep_roi_bgr.empty());
  EXPECT_TRUE(!empty_result.face.has_value());
  EXPECT_TRUE(!empty_result.used_fallback);

  const FramePacket nonempty{10, 4.0, cv::Mat(200, 200, CV_8UC3, cv::Scalar(9, 8, 7))};
  const auto no_face = processor.process(nonempty);
  EXPECT_EQ(detector_calls, 1);
  EXPECT_TRUE(no_face.roi_bgr.empty());
  EXPECT_TRUE(no_face.deep_roi_bgr.empty());
  EXPECT_TRUE(!no_face.face.has_value());
  EXPECT_TRUE(!no_face.used_fallback);
}

void rejects_an_empty_injected_detector() {
  EXPECT_APP_ERROR(RoiProcessor(FaceDetector{}), ErrorCode::ConfigInvalid);
}

void empty_frame_clears_cached_face_and_forces_the_next_detection() {
  int detector_calls = 0;
  FaceDetector detector = [&detector_calls](const cv::Mat&) {
    ++detector_calls;
    if (detector_calls == 1) {
      return std::vector<FaceBox>{FaceBox{20, 20, 100, 100, 0.5}};
    }
    return std::vector<FaceBox>{FaceBox{60, 50, 80, 90, 0.7}};
  };
  RoiProcessor processor(detector);

  const FramePacket initial{1, 1.0, cv::Mat(200, 200, CV_8UC3, cv::Scalar())};
  EXPECT_TRUE(processor.process(initial).face.has_value());

  const FramePacket empty{2, 2.0, {}};
  const auto empty_result = processor.process(empty);
  EXPECT_TRUE(!empty_result.face.has_value());
  EXPECT_TRUE(empty_result.roi_bgr.empty());

  const FramePacket next{3, 3.0, cv::Mat(200, 200, CV_8UC3, cv::Scalar())};
  const auto redetected = processor.process(next);
  EXPECT_EQ(detector_calls, 2);
  EXPECT_TRUE(redetected.face.has_value());
  EXPECT_TRUE(!redetected.used_fallback);
  if (redetected.face.has_value()) {
    EXPECT_EQ(redetected.face->x, 60);
    EXPECT_EQ(redetected.face->y, 50);
  }
}

void size_change_discards_the_cached_face_and_redetects_immediately() {
  int detector_calls = 0;
  FaceDetector detector = [&detector_calls](const cv::Mat&) {
    ++detector_calls;
    if (detector_calls == 1) {
      return std::vector<FaceBox>{FaceBox{20, 20, 100, 100, 0.5}};
    }
    return std::vector<FaceBox>{FaceBox{10, 10, 80, 60, 0.6}};
  };
  RoiProcessor processor(detector);

  const FramePacket large{1, 1.0, cv::Mat(200, 200, CV_8UC3, cv::Scalar())};
  EXPECT_TRUE(processor.process(large).face.has_value());

  const FramePacket resized{2, 2.0, cv::Mat(80, 100, CV_8UC3, cv::Scalar())};
  const auto redetected = processor.process(resized);
  EXPECT_EQ(detector_calls, 2);
  EXPECT_TRUE(redetected.face.has_value());
  EXPECT_TRUE(!redetected.used_fallback);
  EXPECT_TRUE(!redetected.roi_bgr.empty());
  if (redetected.face.has_value()) {
    EXPECT_EQ(redetected.face->x, 10);
    EXPECT_EQ(redetected.face->y, 10);
    EXPECT_EQ(redetected.face->width, 80);
    EXPECT_EQ(redetected.face->height, 60);
  }
  EXPECT_TRUE(redetected.roi_bgr.cols <= resized.bgr.cols);
  EXPECT_TRUE(redetected.roi_bgr.rows <= resized.bgr.rows);
}

void detector_exception_leaves_the_next_valid_frame_due_for_detection() {
  int detector_calls = 0;
  FaceDetector detector = [&detector_calls](const cv::Mat&) {
    ++detector_calls;
    if (detector_calls == 1) {
      throw std::runtime_error("detector failed");
    }
    return std::vector<FaceBox>{FaceBox{30, 30, 90, 90, 0.9}};
  };
  RoiProcessor processor(detector);
  const FramePacket frame{1, 1.0, cv::Mat(200, 200, CV_8UC3, cv::Scalar())};

  bool threw = false;
  try {
    (void)processor.process(frame);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
  EXPECT_EQ(detector_calls, 1);

  const auto retry = processor.process(frame);
  EXPECT_EQ(detector_calls, 2);
  EXPECT_TRUE(retry.face.has_value());
  EXPECT_TRUE(!retry.used_fallback);
  EXPECT_TRUE(!retry.roi_bgr.empty());
}

}  // namespace

int main() {
  returns_python_compatible_expanded_face_geometry();
  rejects_invalid_expanded_face_geometry();
  process_returns_distinct_cloned_traditional_and_deep_crops();
  returns_lower_central_cheek_region();
  clips_partially_out_of_bounds_faces();
  rejects_empty_invalid_and_overflowing_geometry();
  rejects_unloadable_cascade_with_descriptive_config_error();
  reuses_clipped_largest_face_until_the_next_detection_interval();
  empty_input_and_empty_detection_never_return_a_full_frame();
  rejects_an_empty_injected_detector();
  empty_frame_clears_cached_face_and_forces_the_next_detection();
  size_change_discards_the_cached_face_and_redetects_immediately();
  detector_exception_leaves_the_next_valid_frame_due_for_detection();
  return test_support::finish();
}
