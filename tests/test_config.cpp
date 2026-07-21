#include "rppg_qnn/config.hpp"
#include "rppg_qnn/error.hpp"

#include <filesystem>
#include <string>
#include <vector>

#include "test_support.hpp"

namespace {

using rppg_qnn::AppConfig;
using rppg_qnn::ErrorCode;
using rppg_qnn::parse_config;

std::vector<std::string> args(std::initializer_list<const char*> values) {
  std::vector<std::string> result;
  result.reserve(values.size() + 1);
  result.emplace_back("rppg_qnn_live");
  for (const char* value : values) {
    result.emplace_back(value);
  }
  return result;
}

void expect_defaults(const AppConfig& config) {
  EXPECT_EQ(config.camera, "/dev/video0");
  EXPECT_TRUE(config.video.empty());
  EXPECT_EQ(config.width, 1280);
  EXPECT_EQ(config.height, 720);
  EXPECT_EQ(config.fps, 30.0);
  EXPECT_EQ(config.traditional, "green");
  EXPECT_EQ(config.deep, "disabled");
  EXPECT_EQ(config.backend, "gpu");
  EXPECT_EQ(config.qnn_gpu_library, "libQnnGpu.so");
  EXPECT_EQ(config.opencl_library, "libOpenCL.so");
  EXPECT_EQ(config.output, std::filesystem::path("outputs/session"));
  EXPECT_EQ(config.preflight_only, false);
}

}  // namespace

int main() {
  const auto defaults = parse_config(args({}));
  expect_defaults(defaults);

  const auto camera = parse_config(
      args({"--camera", "/dev/video2", "--fps", "30"}));
  EXPECT_EQ(camera.camera, "/dev/video2");
  EXPECT_EQ(camera.fps, 30.0);

  const auto video = parse_config(
      args({"--video", "sample.mp4", "--output", "outputs/a"}));
  EXPECT_EQ(video.video, "sample.mp4");
  EXPECT_EQ(video.output, std::filesystem::path("outputs/a"));

  const auto all_options = parse_config(args({
      "--camera", "/dev/video3", "--width", "1920", "--height", "1080",
      "--fps", "59.94", "--traditional", "green", "--deep", "fake",
      "--backend", "cpu", "--qnn-gpu-library", "/opt/qnn/libQnnGpu.so",
      "--opencl-library", "/opt/opencl/libOpenCL.so", "--output", "run/out",
      "--preflight-only",
  }));
  EXPECT_EQ(all_options.camera, "/dev/video3");
  EXPECT_EQ(all_options.width, 1920);
  EXPECT_EQ(all_options.height, 1080);
  EXPECT_EQ(all_options.fps, 59.94);
  EXPECT_EQ(all_options.traditional, "green");
  EXPECT_EQ(all_options.deep, "fake");
  EXPECT_EQ(all_options.backend, "cpu");
  EXPECT_EQ(all_options.qnn_gpu_library, "/opt/qnn/libQnnGpu.so");
  EXPECT_EQ(all_options.opencl_library, "/opt/opencl/libOpenCL.so");
  EXPECT_EQ(all_options.output, std::filesystem::path("run/out"));
  EXPECT_EQ(all_options.preflight_only, true);

  const auto preflight = parse_config(args({"--preflight-only"}));
  EXPECT_TRUE(preflight.camera.empty());
  EXPECT_TRUE(preflight.video.empty());
  EXPECT_EQ(preflight.preflight_only, true);

  const auto preflight_with_video = parse_config(
      args({"--preflight-only", "--video", "sample.mp4"}));
  EXPECT_EQ(preflight_with_video.video, "sample.mp4");

  EXPECT_APP_ERROR(
      parse_config(args({"--camera", "/dev/video0", "--video", "sample.mp4"})),
      ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--camera", "/dev/video0", "--backend", "cuda"})),
                   ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--unknown"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--camera"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--camera", ""})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--width", "0"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--height", "-1"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--fps", "0"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--width", "12px"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--fps", "30.0fps"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--fps", " 30"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--traditional", "pos"})),
                   ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--deep", "qnn"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--backend", "cuda"})), ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--camera", "/dev/video0", "--camera", "/dev/video1"})),
                   ErrorCode::ConfigInvalid);
  EXPECT_APP_ERROR(parse_config(args({"--preflight-only", "--preflight-only"})),
                   ErrorCode::ConfigInvalid);

  return test_support::finish();
}
