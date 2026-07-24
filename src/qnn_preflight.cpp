#if defined(__linux__) && !defined(__ANDROID__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "rppg_qnn/qnn_preflight.hpp"

#include <dlfcn.h>

#if defined(__linux__) && !defined(__ANDROID__)
#include <link.h>
#endif

#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif

#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

namespace rppg_qnn {
namespace {

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

std::string canonical_path(const std::string& path) {
  std::error_code error;
  const std::filesystem::path canonical = std::filesystem::canonical(path, error);
  return error ? std::string{} : canonical.string();
}

bool has_directory_component(const std::string& path) {
  return path.find('/') != std::string::npos;
}

#if defined(__APPLE__)
std::string resolve_macos_image(const std::string& attempted_path) {
  const std::string canonical_attempted = canonical_path(attempted_path);
  const std::string requested_name =
      std::filesystem::path(attempted_path).filename().string();
  const std::uint32_t image_count = _dyld_image_count();
  for (std::uint32_t index = 0; index < image_count; ++index) {
    const char* image_name = _dyld_get_image_name(index);
    if (image_name == nullptr) {
      continue;
    }
    const std::string image_path = image_name;
    const std::string canonical_image = canonical_path(image_path);
    if (!canonical_attempted.empty() && canonical_image == canonical_attempted) {
      return canonical_image;
    }
    if (image_path == attempted_path) {
      return canonical_image.empty() ? image_path : canonical_image;
    }
    if (!has_directory_component(attempted_path) &&
        std::filesystem::path(image_path).filename() == requested_name) {
      return canonical_image.empty() ? image_path : canonical_image;
    }
  }
  return canonical_attempted;
}
#endif

std::string resolve_loaded_library_path(void* handle,
                                        const std::string& attempted_path) {
#if defined(__ANDROID__)
  (void)handle;
#endif
#if defined(__linux__) && !defined(__ANDROID__)
  link_map* link_map = nullptr;
  if (dlinfo(handle, RTLD_DI_LINKMAP, &link_map) == 0 && link_map != nullptr &&
      link_map->l_name != nullptr && *link_map->l_name != '\0') {
    const std::string loaded_path = link_map->l_name;
    if (!has_directory_component(loaded_path)) {
      return {};
    }
    const std::string canonical_loaded = canonical_path(loaded_path);
    return canonical_loaded.empty() ? loaded_path : canonical_loaded;
  }
#elif defined(__APPLE__)
  (void)handle;
  const std::string image_path = resolve_macos_image(attempted_path);
  if (!image_path.empty()) {
    return image_path;
  }
#endif

  if (has_directory_component(attempted_path)) {
    return canonical_path(attempted_path);
  }
  return {};
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
  result.attempted_path = path;
  if (path.empty()) {
    result.error = "QNN_LIBRARY_NOT_FOUND: empty library path";
    return result;
  }

  void* raw_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (raw_handle == nullptr) {
    result.error = "QNN_LIBRARY_NOT_FOUND: attempted " + path + ": " +
                   dl_error_or("dlopen failed");
    return result;
  }
  DynamicLibrary library(raw_handle);
  result.resolved_path = resolve_loaded_library_path(raw_handle, path);

  for (const auto& symbol_name : required_symbols) {
    dlerror();
    void* symbol = dlsym(raw_handle, symbol_name.c_str());
    const char* lookup_error = dlerror();
    if (lookup_error != nullptr || symbol == nullptr) {
      result.error = "QNN_API_INCOMPATIBLE: attempted " + path +
                     ", missing symbol " + symbol_name;
      if (lookup_error != nullptr) {
        result.error += " (" + std::string(lookup_error) + ")";
      }
      return result;
    }
  }

  result.loaded = true;
  if (result.resolved_path.empty()) {
    result.error = "QNN_LIBRARY_PATH_UNRESOLVED: attempted " + path;
  }
  return result;
}

PreflightResult run_qnn_preflight(const AppConfig& config) {
  const LibraryProbe qnn = probe_library(config.qnn_gpu_library, {});
  const LibraryProbe opencl = probe_library(config.opencl_library, {"clGetPlatformIDs"});

  PreflightResult result;
  result.qnn_gpu_library = qnn.resolved_path.empty() ? qnn.attempted_path
                                                       : qnn.resolved_path;
  result.opencl_library = opencl.resolved_path.empty() ? opencl.attempted_path
                                                        : opencl.resolved_path;
  result.opencl_available = opencl.loaded;
  result.qnn_gpu_available = qnn.loaded && opencl.loaded;

  append_error(result.error, "QNN GPU", qnn.error);
  append_error(result.error, "OpenCL", opencl.error);
  append_error(result.error, "QNN GPU", "deferred_to_sdk_adapter");
  return result;
}

}  // namespace rppg_qnn
