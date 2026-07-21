#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/error.hpp"

#include <cstdint>
#include <string>

#include "test_support.hpp"

int main() {
  const rppg_qnn::HeartRateResult result;

  EXPECT_EQ(result.schema_version, 1);
  EXPECT_TRUE(result.method.empty());

  rppg_qnn::FrameHealth health{};
  EXPECT_EQ(health.schema_version, 1);
  EXPECT_EQ(health.frame_id, std::uint64_t{0});
  EXPECT_EQ(health.timestamp_sec, 0.0);
  EXPECT_EQ(health.capture_fps, 0.0);
  EXPECT_EQ(health.face_found, false);
  EXPECT_EQ(health.face_confidence, 0.0);
  EXPECT_EQ(health.status, "sampling");

  const rppg_qnn::PreflightResult preflight{};
  EXPECT_EQ(preflight.schema_version, 1);
  EXPECT_EQ(preflight.qnn_gpu_available, false);
  EXPECT_EQ(preflight.opencl_available, false);
  EXPECT_TRUE(preflight.qnn_gpu_library.empty());
  EXPECT_TRUE(preflight.opencl_library.empty());
  EXPECT_TRUE(preflight.error.empty());

  struct ErrorCodeMapping {
    rppg_qnn::ErrorCode code;
    const char* name;
  };
  const ErrorCodeMapping mappings[] = {
      {rppg_qnn::ErrorCode::ConfigInvalid, "CONFIG_INVALID"},
      {rppg_qnn::ErrorCode::CameraOpenFailed, "CAMERA_OPEN_FAILED"},
      {rppg_qnn::ErrorCode::CameraFormatUnsupported,
       "CAMERA_FORMAT_UNSUPPORTED"},
      {rppg_qnn::ErrorCode::LowCaptureFps, "LOW_CAPTURE_FPS"},
      {rppg_qnn::ErrorCode::FaceNotFound, "FACE_NOT_FOUND"},
      {rppg_qnn::ErrorCode::QnnLibraryNotFound, "QNN_LIBRARY_NOT_FOUND"},
      {rppg_qnn::ErrorCode::QnnApiIncompatible, "QNN_API_INCOMPATIBLE"},
      {rppg_qnn::ErrorCode::QnnGpuInitFailed, "QNN_GPU_INIT_FAILED"},
      {rppg_qnn::ErrorCode::ModelManifestInvalid, "MODEL_MANIFEST_INVALID"},
      {rppg_qnn::ErrorCode::ModelLoadFailed, "MODEL_LOAD_FAILED"},
      {rppg_qnn::ErrorCode::InferenceFailed, "INFERENCE_FAILED"},
      {rppg_qnn::ErrorCode::OutputWriteFailed, "OUTPUT_WRITE_FAILED"},
  };
  for (const auto& mapping : mappings) {
    EXPECT_EQ(rppg_qnn::to_string(mapping.code), mapping.name);
  }
  EXPECT_EQ(rppg_qnn::to_string(static_cast<rppg_qnn::ErrorCode>(999)),
            "UNKNOWN_ERROR");

  rppg_qnn::AppError error(rppg_qnn::ErrorCode::ConfigInvalid,
                           "camera and video conflict");
  EXPECT_EQ(error.code(), rppg_qnn::ErrorCode::ConfigInvalid);
  EXPECT_TRUE(std::string(error.what()).find("camera and video conflict") !=
              std::string::npos);

  return test_support::finish();
}
