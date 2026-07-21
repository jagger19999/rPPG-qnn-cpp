#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace rppg_qnn {

enum class ErrorCode {
  ConfigInvalid,
  CameraOpenFailed,
  CameraFormatUnsupported,
  LowCaptureFps,
  FaceNotFound,
  QnnLibraryNotFound,
  QnnApiIncompatible,
  QnnGpuInitFailed,
  ModelManifestInvalid,
  ModelLoadFailed,
  InferenceFailed,
  OutputWriteFailed,
};

class AppError : public std::runtime_error {
 public:
  AppError(ErrorCode code, std::string message);

  ErrorCode code() const noexcept { return code_; }

 private:
  ErrorCode code_;
};

std::string_view to_string(ErrorCode code) noexcept;
int exit_code_for(ErrorCode code) noexcept;
void print_error_line(std::string_view code, std::string_view message);

}  // namespace rppg_qnn
