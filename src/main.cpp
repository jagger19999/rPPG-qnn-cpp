#include "rppg_qnn/config.hpp"
#include "rppg_qnn/deep_runtime.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/frame_source.hpp"
#include "rppg_qnn/pipeline.hpp"
#include "rppg_qnn/roi_processor.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
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
      "/opt/homebrew/opt/opencv@4/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
      "/usr/local/opt/opencv@4/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
  };
  for (const auto& candidate : candidates) {
    std::error_code error;
    if (std::filesystem::is_regular_file(candidate, error)) {
      return candidate;
    }
  }
  return candidates.front();
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
    rppg_qnn::PipelineDependencies dependencies{
        [config] {
          return config.video.empty() ? rppg_qnn::make_camera_source(config)
                                      : rppg_qnn::make_video_source(config.video);
        },
        [] { return std::make_unique<rppg_qnn::RoiProcessor>(cascade_path()); },
        [] { return rppg_qnn::make_fake_deep_runtime(std::chrono::milliseconds(0)); },
        {}};
    rppg_qnn::Pipeline pipeline(std::move(config), std::move(dependencies));
    return pipeline.run();
  } catch (const rppg_qnn::AppError& error) {
    rppg_qnn::print_error_line(rppg_qnn::to_string(error.code()), error.what());
    return rppg_qnn::exit_code_for(error.code());
  } catch (const std::exception& error) {
    rppg_qnn::print_error_line("UNEXPECTED_EXCEPTION", error.what());
    return 1;
  }
}
