#pragma once

#include <filesystem>
#include <string>
#include <vector>

namespace rppg_qnn {

struct AppConfig {
  std::string camera;
  std::string video;
  int width{1280};
  int height{720};
  double fps{30.0};
  std::string traditional{"green"};
  std::string deep{"disabled"};
  std::string backend{"gpu"};
  std::string qnn_gpu_library{"libQnnGpu.so"};
  std::string opencl_library{"libOpenCL.so"};
  std::filesystem::path output{"outputs/session"};
  bool preflight_only{false};
};

AppConfig parse_config(const std::vector<std::string>& args);

}  // namespace rppg_qnn
