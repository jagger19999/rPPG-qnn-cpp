#include "rppg_qnn/config.hpp"
#include "rppg_qnn/deep_runtime.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/frame_source.hpp"
#include "rppg_qnn/pipeline.hpp"
#include "rppg_qnn/roi_processor.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace {

std::filesystem::path cascade_path() {
  const char* configured = std::getenv("RPPG_HAAR_CASCADE");
  if (configured != nullptr && *configured != '\0') {
    return configured;
  }
  const std::vector<std::filesystem::path> candidates{
      "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
      "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
      "/opt/homebrew/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
      "/usr/local/opt/opencv/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
      "/opt/homebrew/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
  };
  for (const auto& candidate : candidates) {
    std::error_code error;
    if (std::filesystem::is_regular_file(candidate, error)) {
      return candidate;
    }
  }
  return candidates.front();
}

int app_exit_code(const rppg_qnn::AppError& error) {
  switch (error.code()) {
    case rppg_qnn::ErrorCode::ConfigInvalid: return 2;
    case rppg_qnn::ErrorCode::CameraOpenFailed: return 3;
    case rppg_qnn::ErrorCode::CameraFormatUnsupported: return 4;
    case rppg_qnn::ErrorCode::LowCaptureFps: return 5;
    case rppg_qnn::ErrorCode::FaceNotFound: return 6;
    case rppg_qnn::ErrorCode::QnnLibraryNotFound: return 7;
    case rppg_qnn::ErrorCode::QnnApiIncompatible: return 8;
    case rppg_qnn::ErrorCode::QnnGpuInitFailed: return 9;
    case rppg_qnn::ErrorCode::ModelManifestInvalid: return 10;
    case rppg_qnn::ErrorCode::ModelLoadFailed: return 11;
    case rppg_qnn::ErrorCode::InferenceFailed: return 12;
    case rppg_qnn::ErrorCode::OutputWriteFailed: return 13;
  }
  return 1;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    std::vector<std::string> args;
    args.reserve(static_cast<std::size_t>(argc));
    for (int index = 0; index < argc; ++index) {
      args.emplace_back(argv[index]);
    }
    rppg_qnn::AppConfig config = rppg_qnn::parse_config(args);
    const std::filesystem::path cascade = cascade_path();
    rppg_qnn::PipelineDependencies dependencies{
        [config] {
          return config.video.empty() ? rppg_qnn::make_camera_source(config)
                                      : rppg_qnn::make_video_source(config.video);
        },
        [cascade] { return std::make_unique<rppg_qnn::RoiProcessor>(cascade); },
        [] { return rppg_qnn::make_fake_deep_runtime(std::chrono::milliseconds(0)); }};
    rppg_qnn::Pipeline pipeline(std::move(config), std::move(dependencies));
    return pipeline.run();
  } catch (const rppg_qnn::AppError& error) {
    std::cerr << rppg_qnn::to_string(error.code()) << ": " << error.what() << '\n';
    return app_exit_code(error);
  } catch (const std::exception& error) {
    std::cerr << "UNEXPECTED_EXCEPTION: " << error.what() << '\n';
    return 1;
  }
}
