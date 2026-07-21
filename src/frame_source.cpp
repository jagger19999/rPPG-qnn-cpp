#include "rppg_qnn/frame_source.hpp"

#include "rppg_qnn/error.hpp"

#include <charconv>
#include <chrono>
#include <cmath>
#include <limits>
#include <string>
#include <system_error>
#include <utility>

#include <opencv2/videoio.hpp>

namespace rppg_qnn {
namespace {

[[noreturn]] void input_open_failed(const std::string& message) {
  throw AppError(ErrorCode::CameraOpenFailed, message);
}

class VideoFrameSource final : public FrameSource {
 public:
  explicit VideoFrameSource(const std::filesystem::path& path) {
    if (!capture_.open(path.string())) {
      input_open_failed("Unable to open video input: " + path.string());
    }

    nominal_fps_ = capture_.get(cv::CAP_PROP_FPS);
    if (!std::isfinite(nominal_fps_) || nominal_fps_ <= 0.0) {
      input_open_failed("Video input has no usable frame rate: " + path.string());
    }
    frame_count_ = capture_.get(cv::CAP_PROP_FRAME_COUNT);
  }

  std::optional<FramePacket> read() override {
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
      if (std::isfinite(frame_count_) && frame_count_ > 0.0 &&
          static_cast<double>(next_frame_id_) >= std::ceil(frame_count_)) {
        eof_ = true;
        return std::nullopt;
      }
      input_open_failed("Video input ended before its reported frame count");
    }

    FramePacket packet;
    packet.frame_id = next_frame_id_;
    packet.timestamp_sec = static_cast<double>(next_frame_id_) / nominal_fps_;
    packet.bgr = frame.clone();
    ++next_frame_id_;
    return packet;
  }

  bool eof() const override { return eof_; }
  double nominal_fps() const override { return nominal_fps_; }

 private:
  cv::VideoCapture capture_;
  std::uint64_t next_frame_id_{0};
  double nominal_fps_{0.0};
  double frame_count_{0.0};
  bool eof_{false};
};

int parse_camera_index(const std::string& camera) {
  constexpr char kPrefix[] = "/dev/video";
  constexpr std::size_t kPrefixLength = sizeof(kPrefix) - 1;
  if (camera.compare(0, kPrefixLength, kPrefix) != 0 ||
      camera.size() == kPrefixLength) {
    input_open_failed("Camera device must be /dev/video followed by an index: " + camera);
  }

  const char* const begin = camera.data() + kPrefixLength;
  const char* const end = camera.data() + camera.size();
  for (const char* character = begin; character != end; ++character) {
    if (*character < '0' || *character > '9') {
      input_open_failed("Camera device must be /dev/video followed by an index: " + camera);
    }
  }

  int index = 0;
  const auto conversion = std::from_chars(begin, end, index);
  if (conversion.ec != std::errc{} || conversion.ptr != end || index < 0) {
    input_open_failed("Camera device must be /dev/video followed by an index: " + camera);
  }
  return index;
}

class CameraFrameSource final : public FrameSource {
 public:
  explicit CameraFrameSource(const AppConfig& config) : nominal_fps_(config.fps) {
    if (config.width <= 0 || config.height <= 0 || !std::isfinite(config.fps) ||
        config.fps <= 0.0) {
      throw AppError(ErrorCode::CameraFormatUnsupported,
                     "Camera width, height, and fps must be positive");
    }

    const int camera_index = parse_camera_index(config.camera);
    if (!capture_.open(camera_index, cv::CAP_V4L2)) {
      input_open_failed("Unable to open camera device: " + config.camera);
    }

    capture_.set(cv::CAP_PROP_FRAME_WIDTH, static_cast<double>(config.width));
    capture_.set(cv::CAP_PROP_FRAME_HEIGHT, static_cast<double>(config.height));
    capture_.set(cv::CAP_PROP_FPS, config.fps);

    const double negotiated_width = capture_.get(cv::CAP_PROP_FRAME_WIDTH);
    const double negotiated_height = capture_.get(cv::CAP_PROP_FRAME_HEIGHT);
    if (!std::isfinite(negotiated_width) || negotiated_width <= 0.0 ||
        !std::isfinite(negotiated_height) || negotiated_height <= 0.0) {
      throw AppError(ErrorCode::CameraFormatUnsupported,
                     "Camera reported an unsupported frame format");
    }

    const double negotiated_fps = capture_.get(cv::CAP_PROP_FPS);
    if (std::isfinite(negotiated_fps) && negotiated_fps > 0.0) {
      nominal_fps_ = negotiated_fps;
    }
  }

  std::optional<FramePacket> read() override {
    cv::Mat frame;
    if (!capture_.read(frame) || frame.empty()) {
      if (!has_read_frame_) {
        input_open_failed("Unable to read first frame from camera");
      }
      return std::nullopt;
    }

    const auto now = std::chrono::steady_clock::now();
    double timestamp_sec = 0.0;
    if (!has_read_frame_) {
      first_frame_time_ = now;
      has_read_frame_ = true;
    } else {
      timestamp_sec = std::chrono::duration<double>(now - first_frame_time_).count();
      if (timestamp_sec <= previous_timestamp_sec_) {
        timestamp_sec = std::nextafter(previous_timestamp_sec_,
                                       std::numeric_limits<double>::infinity());
      }
    }

    FramePacket packet;
    packet.frame_id = next_frame_id_;
    packet.timestamp_sec = timestamp_sec;
    packet.bgr = frame.clone();
    previous_timestamp_sec_ = timestamp_sec;
    ++next_frame_id_;
    return packet;
  }

  bool eof() const override { return false; }
  double nominal_fps() const override { return nominal_fps_; }

 private:
  cv::VideoCapture capture_;
  std::chrono::steady_clock::time_point first_frame_time_{};
  std::uint64_t next_frame_id_{0};
  double nominal_fps_{0.0};
  double previous_timestamp_sec_{0.0};
  bool has_read_frame_{false};
};

}  // namespace

std::unique_ptr<FrameSource> make_video_source(const std::filesystem::path& path) {
  return std::make_unique<VideoFrameSource>(path);
}

std::unique_ptr<FrameSource> make_camera_source(const AppConfig& config) {
  return std::make_unique<CameraFrameSource>(config);
}

}  // namespace rppg_qnn
