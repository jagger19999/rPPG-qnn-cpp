#include "rppg_qnn/qnn_preflight.hpp"

#include <dlfcn.h>

#include <cstdlib>
#include <string>

namespace rppg_qnn {
namespace {

constexpr const char* kDefaultQnnGpuLibrary = "libQnnGpu.so";
constexpr const char* kDefaultOpenClLibrary = "libOpenCL.so";

class DynamicLibrary {
 public:
  explicit DynamicLibrary(void* handle) : handle_(handle) {}
  ~DynamicLibrary() {
    if (handle_ != nullptr) {
      dlclose(handle_);
    }
  }

  DynamicLibrary(const DynamicLibrary&) = delete;
  DynamicLibrary& operator=(const DynamicLibrary&) = delete;

 private:
  void* handle_;
};

std::string dl_error_or(const std::string& fallback) {
  const char* error = dlerror();
  return error == nullptr ? fallback : error;
}

std::string resolved_path_or_attempted(const std::string& attempted, void* symbol) {
  Dl_info info{};
  if (symbol != nullptr && dladdr(symbol, &info) != 0 && info.dli_fname != nullptr) {
    return info.dli_fname;
  }
  return attempted;
}

std::string selected_path(const std::string& configured,
                          const char* default_path,
                          const char* environment_variable) {
  if (configured != default_path) {
    return configured;
  }
  const char* environment_path = std::getenv(environment_variable);
  if (environment_path != nullptr && *environment_path != '\0') {
    return environment_path;
  }
  return default_path;
}

void append_error(std::string& result, const std::string& component,
                  const std::string& error) {
  if (error.empty()) {
    return;
  }
  if (!result.empty()) {
    result += "; ";
  }
  result += component + ": " + error;
}

}  // namespace

LibraryProbe probe_library(const std::string& path,
                           const std::vector<std::string>& required_symbols) {
  LibraryProbe result;
  result.resolved_path = path;
  if (path.empty()) {
    result.error = "QNN_LIBRARY_NOT_FOUND: empty library path";
    return result;
  }

  void* raw_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (raw_handle == nullptr) {
    result.error = "QNN_LIBRARY_NOT_FOUND: " + dl_error_or("dlopen failed");
    return result;
  }
  DynamicLibrary library(raw_handle);

  void* last_symbol = nullptr;
  for (const auto& symbol_name : required_symbols) {
    dlerror();
    void* symbol = dlsym(raw_handle, symbol_name.c_str());
    const char* lookup_error = dlerror();
    if (lookup_error != nullptr || symbol == nullptr) {
      result.error = "QNN_API_INCOMPATIBLE: missing symbol " + symbol_name;
      if (lookup_error != nullptr) {
        result.error += " (" + std::string(lookup_error) + ")";
      }
      return result;
    }
    last_symbol = symbol;
  }

  result.loaded = true;
  result.resolved_path = resolved_path_or_attempted(path, last_symbol);
  return result;
}

PreflightResult run_qnn_preflight(const AppConfig& config) {
  const std::string qnn_path = selected_path(
      config.qnn_gpu_library, kDefaultQnnGpuLibrary, "RPPG_QNN_GPU_LIBRARY");
  const std::string opencl_path = selected_path(
      config.opencl_library, kDefaultOpenClLibrary, "RPPG_OPENCL_LIBRARY");

  const LibraryProbe qnn = probe_library(qnn_path, {});
  const LibraryProbe opencl = probe_library(opencl_path, {"clGetPlatformIDs"});

  PreflightResult result;
  result.qnn_gpu_library = qnn.loaded ? qnn.resolved_path : qnn_path;
  result.opencl_library = opencl.loaded ? opencl.resolved_path : opencl_path;
  result.opencl_available = opencl.loaded;
  result.qnn_gpu_available = qnn.loaded && opencl.loaded;

  append_error(result.error, "QNN GPU", qnn.error);
  append_error(result.error, "OpenCL", opencl.error);
  append_error(result.error, "QNN GPU", "deferred_to_sdk_adapter");
  return result;
}

}  // namespace rppg_qnn
