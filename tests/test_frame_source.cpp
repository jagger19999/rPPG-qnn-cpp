#include "rppg_qnn/config.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/frame_source.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <system_error>

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

  ~TemporaryVideo() noexcept {
    std::error_code error;
    std::filesystem::remove(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

void reads_timestamped_video_frames() {
  TemporaryVideo video;
  auto source = make_video_source(video.path());

  EXPECT_TRUE(std::abs(source->nominal_fps() - 24.0) < 0.1);
  EXPECT_TRUE(!source->eof());

  const auto first_packet = source->read();
  EXPECT_TRUE(first_packet.has_value());
  if (!first_packet.has_value()) {
    return;
  }
  EXPECT_EQ(first_packet->frame_id, 0U);
  EXPECT_TRUE(std::abs(first_packet->timestamp_sec) < 1e-12);
  EXPECT_TRUE(first_packet->bgr.cols == 64 && first_packet->bgr.rows == 48);
  EXPECT_TRUE(!first_packet->bgr.empty());
  const cv::Mat first_frame_snapshot = first_packet->bgr.clone();

  double previous_timestamp = first_packet->timestamp_sec;
  for (std::uint64_t expected_id = 1; expected_id < 12; ++expected_id) {
    const auto packet = source->read();
    EXPECT_TRUE(packet.has_value());
    if (!packet.has_value()) {
      break;
    }
    EXPECT_EQ(packet->frame_id, expected_id);
    EXPECT_TRUE(std::abs(packet->timestamp_sec -
                         static_cast<double>(expected_id) / source->nominal_fps()) <
                1e-12);
    EXPECT_TRUE(packet->timestamp_sec > previous_timestamp);
    EXPECT_TRUE(packet->bgr.cols == 64 && packet->bgr.rows == 48);
    EXPECT_TRUE(!packet->bgr.empty());
    EXPECT_TRUE(cv::mean(packet->bgr)[0] > 0.0 || cv::mean(packet->bgr)[1] > 0.0 ||
                cv::mean(packet->bgr)[2] > 0.0);
    EXPECT_TRUE(!source->eof());
    previous_timestamp = packet->timestamp_sec;
  }

  EXPECT_TRUE(cv::norm(first_packet->bgr, first_frame_snapshot, cv::NORM_INF) == 0.0);

  EXPECT_TRUE(!source->read().has_value());
  EXPECT_TRUE(source->eof());
  EXPECT_TRUE(!source->read().has_value());
  EXPECT_TRUE(source->eof());
}

void removes_temporary_video_without_throwing() {
  std::filesystem::path path;
  {
    TemporaryVideo video;
    path = video.path();
    EXPECT_TRUE(std::filesystem::exists(path));
  }
  EXPECT_TRUE(!std::filesystem::exists(path));
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

bool is_invalid_camera_device_name(const std::string& camera) {
  AppConfig config;
  config.camera = camera;
  try {
    (void)make_camera_source(config);
  } catch (const rppg_qnn::AppError& error) {
    return error.code() == ErrorCode::CameraOpenFailed &&
           std::string(error.what()).rfind("Camera device must be /dev/video", 0) == 0;
  }
  return false;
}

void rejects_symbolic_and_whitespace_camera_suffixes() {
  EXPECT_TRUE(is_invalid_camera_device_name("/dev/video-0"));
  EXPECT_TRUE(is_invalid_camera_device_name("/dev/video+0"));
  EXPECT_TRUE(is_invalid_camera_device_name("/dev/video"));
  EXPECT_TRUE(is_invalid_camera_device_name("/dev/video 0"));
  EXPECT_TRUE(is_invalid_camera_device_name("/dev/video0 "));
  EXPECT_TRUE(is_invalid_camera_device_name("/dev/video999999999999999999999999"));
}

void rejects_nonexistent_camera_device() {
  AppConfig config;
  config.camera = "/dev/video999999";
  EXPECT_APP_ERROR(make_camera_source(config), ErrorCode::CameraOpenFailed);
}

}  // namespace

int main() {
  reads_timestamped_video_frames();
  removes_temporary_video_without_throwing();
  rejects_unopenable_video_input();
  rejects_invalid_camera_device_name();
  rejects_symbolic_and_whitespace_camera_suffixes();
  rejects_nonexistent_camera_device();
  return test_support::finish();
}
