#include "rppg_qnn/error.hpp"

#include <utility>

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
  }

  return "UNKNOWN_ERROR";
}

}  // namespace rppg_qnn
