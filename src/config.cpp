#include "rppg_qnn/config.hpp"

#include "rppg_qnn/error.hpp"

#include <charconv>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>

namespace rppg_qnn {
namespace {

[[noreturn]] void config_invalid(const std::string& message) {
  throw AppError(ErrorCode::ConfigInvalid, message);
}

void mark_seen(std::set<std::string>& seen, const std::string& flag) {
  if (!seen.insert(flag).second) {
    config_invalid("Duplicate option " + flag);
  }
}

const std::string& next_value(const std::vector<std::string>& args,
                              std::size_t& index,
                              const std::string& flag) {
  if (index + 1 >= args.size() || args[index + 1].empty() ||
      args[index + 1].rfind("--", 0) == 0) {
    config_invalid("Option " + flag + " requires a non-empty value");
  }
  ++index;
  return args[index];
}

int positive_int(const std::string& value, const std::string& flag) {
  int result = 0;
  const char* const begin = value.data();
  const char* const end = begin + value.size();
  const auto conversion = std::from_chars(begin, end, result);
  if (conversion.ec != std::errc{} || conversion.ptr != end || result <= 0) {
    config_invalid("Invalid value for " + flag + ": expected a positive integer");
  }
  return result;
}

double positive_double(const std::string& value, const std::string& flag) {
  for (const unsigned char character : value) {
    if (std::isspace(character) != 0) {
      config_invalid("Invalid value for " + flag + ": expected a positive number");
    }
  }

  std::size_t consumed = 0;
  double result = 0.0;
  try {
    result = std::stod(value, &consumed);
  } catch (const std::invalid_argument&) {
    config_invalid("Invalid value for " + flag + ": expected a positive number");
  } catch (const std::out_of_range&) {
    config_invalid("Invalid value for " + flag + ": expected a positive number");
  }

  if (consumed != value.size() || !std::isfinite(result) || result <= 0.0) {
    config_invalid("Invalid value for " + flag + ": expected a positive number");
  }
  return result;
}

void require_one_of(const std::string& value,
                    const std::string& flag,
                    const std::string& first,
                    const std::string& second = "") {
  if (value == first || (!second.empty() && value == second)) {
    return;
  }

  std::string message = "Invalid value for " + flag + ": expected " + first;
  if (!second.empty()) {
    message += " or " + second;
  }
  config_invalid(message);
}

void apply_environment_default(std::string& configured_value,
                               const char* environment_variable) {
  const char* environment_value = std::getenv(environment_variable);
  if (environment_value != nullptr && *environment_value != '\0') {
    configured_value = environment_value;
  }
}

}  // namespace

AppConfig parse_config(const std::vector<std::string>& args) {
  AppConfig config;
  apply_environment_default(config.qnn_gpu_library, "RPPG_QNN_GPU_LIBRARY");
  apply_environment_default(config.opencl_library, "RPPG_OPENCL_LIBRARY");
  std::set<std::string> seen;

  for (std::size_t index = 1; index < args.size(); ++index) {
    const std::string& flag = args[index];

    if (flag == "--preflight-only") {
      mark_seen(seen, flag);
      config.preflight_only = true;
    } else if (flag == "--camera") {
      mark_seen(seen, flag);
      config.camera = next_value(args, index, flag);
    } else if (flag == "--video") {
      mark_seen(seen, flag);
      config.video = next_value(args, index, flag);
    } else if (flag == "--width") {
      mark_seen(seen, flag);
      config.width = positive_int(next_value(args, index, flag), flag);
    } else if (flag == "--height") {
      mark_seen(seen, flag);
      config.height = positive_int(next_value(args, index, flag), flag);
    } else if (flag == "--fps") {
      mark_seen(seen, flag);
      config.fps = positive_double(next_value(args, index, flag), flag);
    } else if (flag == "--traditional") {
      mark_seen(seen, flag);
      config.traditional = next_value(args, index, flag);
      require_one_of(config.traditional, flag, "green");
    } else if (flag == "--deep") {
      mark_seen(seen, flag);
      config.deep = next_value(args, index, flag);
      require_one_of(config.deep, flag, "disabled", "fake");
    } else if (flag == "--backend") {
      mark_seen(seen, flag);
      config.backend = next_value(args, index, flag);
      require_one_of(config.backend, flag, "gpu", "cpu");
    } else if (flag == "--qnn-gpu-library") {
      mark_seen(seen, flag);
      config.qnn_gpu_library = next_value(args, index, flag);
    } else if (flag == "--opencl-library") {
      mark_seen(seen, flag);
      config.opencl_library = next_value(args, index, flag);
    } else if (flag == "--output") {
      mark_seen(seen, flag);
      config.output = next_value(args, index, flag);
    } else {
      config_invalid("Unknown option " + flag);
    }
  }

  if (!config.camera.empty() && !config.video.empty()) {
    config_invalid("--camera and --video cannot be used together");
  }
  if (config.camera.empty() && config.video.empty() && !config.preflight_only) {
    config.camera = "/dev/video0";
  }

  return config;
}

}  // namespace rppg_qnn
