#include "rppg_qnn/error.hpp"

#include <utility>

#include <iostream>
#include <mutex>

namespace rppg_qnn {

AppError::AppError(ErrorCode code, std::string message)
    : std::runtime_error(std::move(message)), code_(code) {}

std::string_view to_string(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::ConfigInvalid:
      return "CONFIG_INVALID";
    case ErrorCode::CameraOpenFailed:
      return "CAMERA_OPEN_FAILED";
    case ErrorCode::CameraFormatUnsupported:
      return "CAMERA_FORMAT_UNSUPPORTED";
    case ErrorCode::LowCaptureFps:
      return "LOW_CAPTURE_FPS";
    case ErrorCode::FaceNotFound:
      return "FACE_NOT_FOUND";
    case ErrorCode::QnnLibraryNotFound:
      return "QNN_LIBRARY_NOT_FOUND";
    case ErrorCode::QnnApiIncompatible:
      return "QNN_API_INCOMPATIBLE";
    case ErrorCode::QnnGpuInitFailed:
      return "QNN_GPU_INIT_FAILED";
    case ErrorCode::ModelManifestInvalid:
      return "MODEL_MANIFEST_INVALID";
    case ErrorCode::ModelLoadFailed:
      return "MODEL_LOAD_FAILED";
    case ErrorCode::InferenceFailed:
      return "INFERENCE_FAILED";
    case ErrorCode::OutputWriteFailed:
      return "OUTPUT_WRITE_FAILED";
    case ErrorCode::CameraPermissionDenied:
      return "CAMERA_PERMISSION_DENIED";
    case ErrorCode::CameraIdUnavailable:
      return "CAMERA_ID_UNAVAILABLE";
    case ErrorCode::CameraImageInvalid:
      return "CAMERA_IMAGE_INVALID";
    case ErrorCode::NativeStateInvalid:
      return "NATIVE_STATE_INVALID";
  }

  return "UNKNOWN_ERROR";
}

int exit_code_for(ErrorCode code) noexcept {
  switch (code) {
    case ErrorCode::ConfigInvalid: return 2;
    case ErrorCode::CameraOpenFailed: return 3;
    case ErrorCode::CameraFormatUnsupported: return 4;
    case ErrorCode::LowCaptureFps: return 5;
    case ErrorCode::FaceNotFound: return 6;
    case ErrorCode::QnnLibraryNotFound: return 7;
    case ErrorCode::QnnApiIncompatible: return 8;
    case ErrorCode::QnnGpuInitFailed: return 9;
    case ErrorCode::ModelManifestInvalid: return 10;
    case ErrorCode::ModelLoadFailed: return 11;
    case ErrorCode::InferenceFailed: return 12;
    case ErrorCode::OutputWriteFailed: return 13;
    case ErrorCode::CameraPermissionDenied: return 14;
    case ErrorCode::CameraIdUnavailable: return 15;
    case ErrorCode::CameraImageInvalid: return 16;
    case ErrorCode::NativeStateInvalid: return 17;
  }
  return 1;
}

void print_error_line(std::string_view code, std::string_view message) {
  static std::mutex mutex;
  std::lock_guard<std::mutex> lock(mutex);
  std::cerr << code << ": " << message << '\n';
}

}  // namespace rppg_qnn
