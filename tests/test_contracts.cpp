#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/error.hpp"

#include <cstdint>
#include <string>

#include "test_support.hpp"

int main() {
  const rppg_qnn::HeartRateResult result;

  EXPECT_EQ(result.schema_version, 1);
  EXPECT_TRUE(result.method.empty());
  EXPECT_EQ(result.peak_ratio, 0.0);

  const rppg_qnn::QualityProfile profile;
  EXPECT_EQ(profile.schema_version, 1);
  EXPECT_EQ(profile.min_brightness, 0.10);
  EXPECT_EQ(profile.max_brightness, 0.92);
  EXPECT_EQ(profile.min_face_area_ratio, 0.015);
  EXPECT_EQ(profile.max_motion_px, 32.0);
  EXPECT_EQ(profile.max_bpm_jump, 18.0);

  const rppg_qnn::MeasurementSnapshot snapshot;
  EXPECT_EQ(snapshot.schema_version, 1);
  EXPECT_TRUE(snapshot.candidates.empty());
  EXPECT_TRUE(!snapshot.gate.accepted);
  EXPECT_TRUE(!snapshot.display_available);
  EXPECT_TRUE(!snapshot.display_is_held);

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
    int exit_code;
  };
  const ErrorCodeMapping mappings[] = {
      {rppg_qnn::ErrorCode::ConfigInvalid, "CONFIG_INVALID", 2},
      {rppg_qnn::ErrorCode::CameraOpenFailed, "CAMERA_OPEN_FAILED", 3},
      {rppg_qnn::ErrorCode::CameraFormatUnsupported,
       "CAMERA_FORMAT_UNSUPPORTED", 4},
      {rppg_qnn::ErrorCode::LowCaptureFps, "LOW_CAPTURE_FPS", 5},
      {rppg_qnn::ErrorCode::FaceNotFound, "FACE_NOT_FOUND", 6},
      {rppg_qnn::ErrorCode::QnnLibraryNotFound, "QNN_LIBRARY_NOT_FOUND", 7},
      {rppg_qnn::ErrorCode::QnnApiIncompatible, "QNN_API_INCOMPATIBLE", 8},
      {rppg_qnn::ErrorCode::QnnGpuInitFailed, "QNN_GPU_INIT_FAILED", 9},
      {rppg_qnn::ErrorCode::ModelManifestInvalid, "MODEL_MANIFEST_INVALID", 10},
      {rppg_qnn::ErrorCode::ModelLoadFailed, "MODEL_LOAD_FAILED", 11},
      {rppg_qnn::ErrorCode::InferenceFailed, "INFERENCE_FAILED", 12},
      {rppg_qnn::ErrorCode::OutputWriteFailed, "OUTPUT_WRITE_FAILED", 13},
      {rppg_qnn::ErrorCode::CameraPermissionDenied,
       "CAMERA_PERMISSION_DENIED", 14},
      {rppg_qnn::ErrorCode::CameraIdUnavailable, "CAMERA_ID_UNAVAILABLE", 15},
      {rppg_qnn::ErrorCode::CameraImageInvalid, "CAMERA_IMAGE_INVALID", 16},
      {rppg_qnn::ErrorCode::NativeStateInvalid, "NATIVE_STATE_INVALID", 17},
  };
  for (const auto& mapping : mappings) {
    EXPECT_EQ(rppg_qnn::to_string(mapping.code), mapping.name);
    EXPECT_EQ(rppg_qnn::exit_code_for(mapping.code), mapping.exit_code);
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
