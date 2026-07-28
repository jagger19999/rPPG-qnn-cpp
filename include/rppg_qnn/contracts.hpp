#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace rppg_qnn {

struct TimingDistribution {
  double latest_ms{0.0};
  double p50_ms{0.0};
  double p95_ms{0.0};
};

struct PerformanceMetrics {
  double processing_fps{0.0};
  TimingDistribution roi;
  TimingDistribution traditional;
  TimingDistribution deep_preprocess;
};

struct FaceBox {
  int x{0};
  int y{0};
  int width{0};
  int height{0};
  double confidence{0.0};
};

struct RoiPacket {
  RoiPacket() = default;
  RoiPacket(std::uint64_t requested_frame_id, double requested_timestamp_sec,
            cv::Mat requested_roi_bgr, std::optional<FaceBox> requested_face,
            bool requested_used_fallback, int requested_face_count = 0,
            double requested_motion_px = 0.0,
            cv::Mat requested_deep_roi_bgr = {})
      : frame_id(requested_frame_id),
        timestamp_sec(requested_timestamp_sec),
        roi_bgr(std::move(requested_roi_bgr)),
        face(std::move(requested_face)),
        used_fallback(requested_used_fallback),
        face_count(requested_face_count),
        motion_px(requested_motion_px),
        deep_roi_bgr(std::move(requested_deep_roi_bgr)) {}
  RoiPacket(std::uint64_t requested_frame_id, double requested_timestamp_sec,
            cv::Mat requested_roi_bgr, std::optional<FaceBox> requested_face,
            bool requested_used_fallback, cv::Mat requested_deep_roi_bgr)
      : RoiPacket(requested_frame_id, requested_timestamp_sec,
                  std::move(requested_roi_bgr), std::move(requested_face),
                  requested_used_fallback, 0, 0.0,
                  std::move(requested_deep_roi_bgr)) {}
  std::uint64_t frame_id{0};
  double timestamp_sec{0.0};
  cv::Mat roi_bgr;
  std::optional<FaceBox> face;
  bool used_fallback{false};
  int face_count{0};
  double motion_px{0.0};
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
  PerformanceMetrics performance;
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
  double peak_ratio{0.0};
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

struct QualityProfile {
  int schema_version{1};
  double min_brightness{0.10};
  double max_brightness{0.92};
  double min_signal_std{0.003};
  double min_peak_ratio{0.08};
  double min_face_area_ratio{0.015};
  double max_motion_px{32.0};
  double max_bpm_jump{18.0};
  double min_source_fps{15.0};
  double max_frame_gap_sec{0.75};
  double max_method_spread_bpm{12.0};
  double hold_last_reliable_sec{5.0};
};

struct QualityMetrics {
  double brightness{0.0};
  double brightness_std{0.0};
  double signal_std{0.0};
  double spectral_peak_ratio{0.0};
  double face_area_ratio{0.0};
  double motion_px{0.0};
  double source_fps{0.0};
  double max_frame_gap_sec{0.0};
  int face_count{0};
};

struct CandidateResult {
  std::string method;
  double bpm{0.0};
  double confidence{0.0};
  double peak_ratio{0.0};
  bool valid{false};
  std::string invalid_reason;
  std::vector<float> waveform;
};

struct GateDecision {
  bool accepted{false};
  double quality_score{0.0};
  std::vector<std::string> flags;
  std::string reason{"sampling"};
};

struct VqaAssessment {
  double score_0_10{0.0};
  std::string label{"Poor"};
  std::vector<std::string> flags;
  double head_movement_score{0.0};
  double illumination_score{0.0};
  double skin_score{0.0};
  double camera_score{0.0};
  double method_consensus_score{0.0};
};

struct RouterShadowDecision {
  bool active{true};
  bool model_loaded{false};
  std::string backend{"heuristic_shadow"};
  std::string recommended_method;
  double gate_probability{0.0};
  double evaluation_ms{0.0};
};

struct MeasurementSnapshot {
  int schema_version{1};
  double window_end_sec{0.0};
  std::vector<CandidateResult> candidates;
  QualityMetrics quality;
  GateDecision gate;
  VqaAssessment vqa;
  RouterShadowDecision router_shadow;
  double measured_bpm{0.0};
  double accepted_bpm{0.0};
  double display_bpm{0.0};
  bool measured_available{false};
  bool accepted_available{false};
  bool display_available{false};
  bool display_is_held{false};
  std::string selected_method;
  std::vector<std::string> consensus_methods;
  double consensus_spread_bpm{0.0};
  std::string consensus_artifact_correction;
  std::string consensus_rejection_reason;
};

}  // namespace rppg_qnn
