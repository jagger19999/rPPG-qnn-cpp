#if defined(RPPG_FAKE_OPENCL_LIBRARY)

extern "C" void clGetPlatformIDs() {}

#else

#include "rppg_qnn/config.hpp"
#include "rppg_qnn/qnn_preflight.hpp"

#include <dlfcn.h>
#include <cstdlib>
#include <string>

#include "test_support.hpp"

namespace {

using rppg_qnn::AppConfig;
using rppg_qnn::parse_config;
using rppg_qnn::probe_library;
using rppg_qnn::run_qnn_preflight;

bool contains(const std::string& value, const std::string& fragment) {
  return value.find(fragment) != std::string::npos;
}

std::string platform_c_library() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<void*>(dlopen), &info) == 0 ||
      info.dli_fname == nullptr) {
    return {};
  }
  return info.dli_fname;
}

class EnvironmentValue {
 public:
  explicit EnvironmentValue(const char* name) : name_(name) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      had_value_ = true;
      value_ = value;
    }
  }

  ~EnvironmentValue() {
    if (had_value_) {
      setenv(name_.c_str(), value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  bool had_value_{false};
  std::string value_;
};

}  // namespace

int main() {
  const std::string c_library = platform_c_library();
  EXPECT_TRUE(!c_library.empty());

  const auto empty = probe_library("", {});
  EXPECT_EQ(empty.loaded, false);
  EXPECT_TRUE(contains(empty.error, "QNN_LIBRARY_NOT_FOUND"));

  const auto missing = probe_library("/definitely/not/a/qnn/library.so", {});
  EXPECT_EQ(missing.loaded, false);
  EXPECT_TRUE(contains(missing.error, "QNN_LIBRARY_NOT_FOUND"));

  const auto c_library_probe = probe_library(c_library, {"dlopen"});
  EXPECT_TRUE(c_library_probe.loaded);
  EXPECT_EQ(c_library_probe.resolved_path, c_library);
  EXPECT_TRUE(c_library_probe.error.empty());

  const auto incompatible = probe_library(c_library, {"rppg_qnn_symbol_that_does_not_exist"});
  EXPECT_EQ(incompatible.loaded, false);
  EXPECT_TRUE(contains(incompatible.error, "QNN_API_INCOMPATIBLE"));
  EXPECT_TRUE(contains(incompatible.error, "rppg_qnn_symbol_that_does_not_exist"));

  EnvironmentValue qnn_environment("RPPG_QNN_GPU_LIBRARY");
  EnvironmentValue opencl_environment("RPPG_OPENCL_LIBRARY");
  setenv("RPPG_QNN_GPU_LIBRARY", "/environment/qnn.so", 1);
  setenv("RPPG_OPENCL_LIBRARY", "/environment/opencl.so", 1);

  const AppConfig explicit_config = parse_config({
      "rppg_qnn_live", "--preflight-only", "--qnn-gpu-library", "/cli/qnn.so",
      "--opencl-library", "/cli/opencl.so",
  });
  const auto explicit_result = run_qnn_preflight(explicit_config);
  EXPECT_EQ(explicit_result.qnn_gpu_library, "/cli/qnn.so");
  EXPECT_EQ(explicit_result.opencl_library, "/cli/opencl.so");

  AppConfig environment_config;
  const auto environment_result = run_qnn_preflight(environment_config);
  EXPECT_EQ(environment_result.qnn_gpu_library, "/environment/qnn.so");
  EXPECT_EQ(environment_result.opencl_library, "/environment/opencl.so");
  EXPECT_TRUE(contains(environment_result.error, "QNN_LIBRARY_NOT_FOUND"));

  unsetenv("RPPG_QNN_GPU_LIBRARY");
  unsetenv("RPPG_OPENCL_LIBRARY");
  const auto default_result = run_qnn_preflight(AppConfig{});
  EXPECT_EQ(default_result.qnn_gpu_library, "libQnnGpu.so");
  EXPECT_EQ(default_result.opencl_library, "libOpenCL.so");
  EXPECT_EQ(default_result.qnn_gpu_available, false);
  EXPECT_TRUE(contains(default_result.error, "QNN_LIBRARY_NOT_FOUND"));

  AppConfig opencl_missing_symbol_config;
  opencl_missing_symbol_config.qnn_gpu_library = c_library;
  opencl_missing_symbol_config.opencl_library = c_library;
  const auto opencl_missing_symbol = run_qnn_preflight(opencl_missing_symbol_config);
  EXPECT_EQ(opencl_missing_symbol.opencl_available, false);
  EXPECT_EQ(opencl_missing_symbol.qnn_gpu_available, false);
  EXPECT_TRUE(contains(opencl_missing_symbol.error, "QNN_API_INCOMPATIBLE"));

  AppConfig mixed_failure_config;
  mixed_failure_config.qnn_gpu_library = "/definitely/not/a/qnn/library.so";
  mixed_failure_config.opencl_library = c_library;
  const auto mixed_failure = run_qnn_preflight(mixed_failure_config);
  EXPECT_EQ(mixed_failure.qnn_gpu_available, false);
  EXPECT_TRUE(contains(mixed_failure.error, "QNN_LIBRARY_NOT_FOUND"));
  EXPECT_TRUE(contains(mixed_failure.error, "QNN_API_INCOMPATIBLE"));
  EXPECT_TRUE(contains(mixed_failure.error, "QNN GPU"));
  EXPECT_TRUE(contains(mixed_failure.error, "OpenCL"));

  AppConfig readiness_config;
  readiness_config.qnn_gpu_library = c_library;
  readiness_config.opencl_library = RPPG_FAKE_OPENCL_LIBRARY_PATH;
  const auto readiness = run_qnn_preflight(readiness_config);
  EXPECT_TRUE(readiness.opencl_available);
  EXPECT_TRUE(readiness.qnn_gpu_available);
  EXPECT_TRUE(contains(readiness.error, "deferred_to_sdk_adapter"));

  return test_support::finish();
}

#endif
