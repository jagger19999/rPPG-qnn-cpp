#include "rppg_qnn/config.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/frame_source.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include "test_support.hpp"

namespace {

using rppg_qnn::AppConfig;
using rppg_qnn::ErrorCode;
using rppg_qnn::make_camera_source;
using rppg_qnn::make_video_source;

class TemporaryVideo {
 public:
  TemporaryVideo()
      : path_(std::filesystem::temp_directory_path() /
              ("rppg_qnn_frame_source_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch()
                                  .count()) +
               ".avi")) {
    cv::VideoWriter writer(path_.string(),
                           cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                           24.0,
                           cv::Size(64, 48));
    EXPECT_TRUE(writer.isOpened());
    for (int index = 0; index < 12; ++index) {
      writer.write(cv::Mat(48, 64, CV_8UC3,
                           cv::Scalar(index * 10, 40 + index, 180 - index)));
    }
  }

  ~TemporaryVideo() { std::filesystem::remove(path_); }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void reads_timestamped_video_frames() {
  TemporaryVideo video;
  auto source = make_video_source(video.path());

  EXPECT_TRUE(std::abs(source->nominal_fps() - 24.0) < 0.1);

  double previous_timestamp = -1.0;
  for (std::uint64_t expected_id = 0; expected_id < 12; ++expected_id) {
    const auto packet = source->read();
    EXPECT_TRUE(packet.has_value());
    if (!packet.has_value()) {
      break;
    }
    EXPECT_EQ(packet->frame_id, expected_id);
    EXPECT_TRUE(packet->timestamp_sec > previous_timestamp);
    EXPECT_TRUE(packet->bgr.cols == 64 && packet->bgr.rows == 48);
    EXPECT_TRUE(!packet->bgr.empty());
    EXPECT_TRUE(cv::mean(packet->bgr)[0] > 0.0 || cv::mean(packet->bgr)[1] > 0.0 ||
                cv::mean(packet->bgr)[2] > 0.0);
    previous_timestamp = packet->timestamp_sec;
  }

  EXPECT_TRUE(!source->read().has_value());
  EXPECT_TRUE(source->eof());
}

void rejects_unopenable_video_input() {
  EXPECT_APP_ERROR(make_video_source("/definitely/missing/rppg_input.avi"),
                   ErrorCode::CameraOpenFailed);
}

void rejects_invalid_camera_device_name() {
  AppConfig config;
  config.camera = "/dev/video12x";
  EXPECT_APP_ERROR(make_camera_source(config), ErrorCode::CameraOpenFailed);
}

void rejects_nonexistent_camera_device() {
  AppConfig config;
  config.camera = "/dev/video999999";
  EXPECT_APP_ERROR(make_camera_source(config), ErrorCode::CameraOpenFailed);
}

}  // namespace

int main() {
  reads_timestamped_video_frames();
  rejects_unopenable_video_input();
  rejects_invalid_camera_device_name();
  rejects_nonexistent_camera_device();
  return test_support::finish();
}
