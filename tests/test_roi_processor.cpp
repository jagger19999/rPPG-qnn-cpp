#include "rppg_qnn/error.hpp"
#include "rppg_qnn/roi_processor.hpp"

#include <filesystem>
#include <limits>
#include <string>

#include <opencv2/core.hpp>

#include "test_support.hpp"

namespace {

using rppg_qnn::AppError;
using rppg_qnn::ErrorCode;
using rppg_qnn::FaceBox;
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

}  // namespace

int main() {
  returns_lower_central_cheek_region();
  clips_partially_out_of_bounds_faces();
  rejects_empty_invalid_and_overflowing_geometry();
  rejects_unloadable_cascade_with_descriptive_config_error();
  return test_support::finish();
}
