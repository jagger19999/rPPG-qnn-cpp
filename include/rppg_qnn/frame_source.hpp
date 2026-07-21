#pragma once

#include "rppg_qnn/config.hpp"

#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>

#include <opencv2/core.hpp>

namespace rppg_qnn {

struct FramePacket {
  std::uint64_t frame_id{0};
  double timestamp_sec{0};
  cv::Mat bgr;
};

class FrameSource {
 public:
  virtual ~FrameSource() = default;

  virtual std::optional<FramePacket> read() = 0;
  virtual bool eof() const = 0;
  virtual double nominal_fps() const = 0;
};

std::unique_ptr<FrameSource> make_video_source(const std::filesystem::path& path);
std::unique_ptr<FrameSource> make_camera_source(const AppConfig& config);

}  // namespace rppg_qnn
