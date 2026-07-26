#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace rppg_qnn {

struct FaceBox {
  int x{0};
  int y{0};
  int width{0};
  int height{0};
  double confidence{0.0};
};

struct RoiPacket {
  std::uint64_t frame_id{0};
  double timestamp_sec{0.0};
  cv::Mat roi_bgr;
  std::optional<FaceBox> face;
  bool used_fallback{false};
  cv::Mat deep_roi_bgr;
};

struct FrameHealth {
  int schema_version{1};
  std::uint64_t frame_id{0};
  double timestamp_sec{0.0};
  double capture_fps{0.0};
  bool face_found{false};
  double face_confidence{0.0};
  std::string status{"sampling"};
};

struct PreflightResult {
  int schema_version{1};
  bool qnn_gpu_available{false};
  bool opencl_available{false};
  std::string qnn_gpu_library;
  std::string opencl_library;
  std::string error;
};

struct HeartRateResult {
  int schema_version{1};
  std::string method;
  double window_start_sec{0.0};
  double window_end_sec{0.0};
  double bpm{0.0};
  double raw_bpm{0.0};
  double display_bpm{0.0};
  double confidence{0.0};
  bool is_valid{false};
  bool stability_valid{false};
  std::string correction_reason;
  std::string invalid_reason;
  double source_fps{0.0};
  std::size_t source_frame_count{0};
  double max_frame_gap_sec{0.0};
  double window_materialization_ms{0.0};
  double preprocess_ms{0.0};
  double runtime_ms{0.0};
  double postprocess_ms{0.0};
  double inference_ms{0.0};
  std::string backend;
  std::string model_sha256;
  std::vector<float> waveform;
};

}  // namespace rppg_qnn
