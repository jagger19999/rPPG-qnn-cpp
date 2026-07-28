#include "android_jni_handle.hpp"

#include "android_camera_session.hpp"
#include "rppg_qnn/error.hpp"

#include <android/native_window.h>

#include <atomic>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rppg_qnn::android {
namespace {

std::mutex registry_mutex;
std::unordered_map<std::int64_t, std::shared_ptr<AndroidCameraSession>> registry;
std::atomic<std::int64_t> next_handle{1};

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (const unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec;
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

std::shared_ptr<AndroidCameraSession> lookup(std::int64_t handle) {
  if (handle <= 0) {
    throw AppError(ErrorCode::NativeStateInvalid,
                   "camera session handle must be positive");
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  const auto found = registry.find(handle);
  if (found == registry.end()) {
    throw AppError(ErrorCode::NativeStateInvalid,
                   "camera session handle is not active");
  }
  return found->second;
}

std::string measurement_json(const MeasurementSnapshot& measurement) {
  std::ostringstream output;
  output << std::setprecision(10)
         << "{\"measured_available\":"
         << (measurement.measured_available ? "true" : "false")
         << ",\"measured_bpm\":" << measurement.measured_bpm
         << ",\"accepted_available\":"
         << (measurement.accepted_available ? "true" : "false")
         << ",\"accepted_bpm\":" << measurement.accepted_bpm
         << ",\"display_available\":"
         << (measurement.display_available ? "true" : "false")
         << ",\"display_bpm\":" << measurement.display_bpm
         << ",\"display_is_held\":"
         << (measurement.display_is_held ? "true" : "false")
         << ",\"selected_method\":\""
         << json_escape(measurement.selected_method)
         << "\",\"quality_score\":" << measurement.gate.quality_score
         << ",\"gate_accepted\":"
         << (measurement.gate.accepted ? "true" : "false")
         << ",\"gate_reason\":\"" << json_escape(measurement.gate.reason)
         << "\",\"vqa_score\":" << measurement.vqa.score_0_10
         << ",\"vqa_label\":\"" << json_escape(measurement.vqa.label)
         << "\",\"router_method\":\""
         << json_escape(measurement.router_shadow.recommended_method)
         << "\",\"router_gate_probability\":"
         << measurement.router_shadow.gate_probability
         << ",\"router_backend\":\""
         << json_escape(measurement.router_shadow.backend)
         << "\",\"router_model_loaded\":"
         << (measurement.router_shadow.model_loaded ? "true" : "false")
         << ",\"consensus_spread_bpm\":"
         << measurement.consensus_spread_bpm
         << ",\"consensus_artifact_correction\":\""
         << json_escape(measurement.consensus_artifact_correction)
         << "\",\"consensus_rejection_reason\":\""
         << json_escape(measurement.consensus_rejection_reason)
         << "\",\"consensus_methods\":[";
  for (std::size_t index = 0; index < measurement.consensus_methods.size();
       ++index) {
    if (index != 0U) output << ',';
    output << "\"" << json_escape(measurement.consensus_methods[index]) << "\"";
  }
  output << "],\"flags\":[";
  for (std::size_t index = 0; index < measurement.gate.flags.size(); ++index) {
    if (index != 0U) output << ',';
    output << "\"" << json_escape(measurement.gate.flags[index]) << "\"";
  }
  output << "],\"candidates\":[";
  for (std::size_t index = 0; index < measurement.candidates.size(); ++index) {
    if (index != 0U) output << ',';
    const CandidateResult& candidate = measurement.candidates[index];
    output << "{\"method\":\"" << json_escape(candidate.method)
           << "\",\"bpm\":" << candidate.bpm
           << ",\"confidence\":" << candidate.confidence
           << ",\"peak_ratio\":" << candidate.peak_ratio
           << ",\"valid\":" << (candidate.valid ? "true" : "false")
           << ",\"invalid_reason\":\""
           << json_escape(candidate.invalid_reason) << "\"}";
  }
  output << "]}";
  return output.str();
}

std::string status_json(const CameraSessionStatus& status) {
  std::ostringstream output;
  output << std::setprecision(10)
         << "{\"state\":\"" << json_escape(status.state)
         << "\",\"camera_id\":\"" << json_escape(status.camera_id)
         << "\",\"requested_width\":" << status.requested_width
         << ",\"requested_height\":" << status.requested_height
         << ",\"requested_fps\":" << status.requested_fps
         << ",\"target_fps_min\":" << status.target_fps_min
         << ",\"target_fps_max\":" << status.target_fps_max
         << ",\"sensor_orientation\":" << status.sensor_orientation
         << ",\"display_rotation\":" << status.display_rotation
         << ",\"frame_rotation\":" << status.frame_rotation
         << ",\"measured_fps\":" << status.measured_fps
         << ",\"processing_fps\":" << status.performance.processing_fps
         << ",\"roi_p95_ms\":" << status.performance.roi.p95_ms
         << ",\"traditional_p95_ms\":"
         << status.performance.traditional.p95_ms
         << ",\"deep_preprocess_p95_ms\":"
         << status.performance.deep_preprocess.p95_ms
         << ",\"color_diagnostic_state\":\""
         << json_escape(status.color_diagnostic_state)
         << "\",\"color_diagnostic_path\":\""
         << json_escape(status.color_diagnostic_path)
         << "\",\"color_diagnostic_finding\":\""
         << json_escape(status.color_diagnostic_finding) << "\""
         << ",\"accepted_frames\":" << status.accepted_frames
         << ",\"dropped_frames\":" << status.dropped_frames
         << ",\"last_timestamp_sec\":" << status.last_timestamp_sec
         << ",\"processing_enabled\":"
         << (status.processing_enabled ? "true" : "false")
         << ",\"face_found\":" << (status.face_found ? "true" : "false");
  if (status.face_rect_available) {
    output << ",\"face_rect\":{\"x\":" << status.face_rect_x
           << ",\"y\":" << status.face_rect_y << ",\"w\":" << status.face_rect_w
           << ",\"h\":" << status.face_rect_h << "}";
  }
  output << ",\"preview_enabled\":" << (status.preview_enabled ? "true" : "false")
         << ",\"traditional_method\":\""
         << json_escape(status.traditional_method)
         << "\",\"heart_rate_available\":"
         << (status.heart_rate_available ? "true" : "false")
         << ",\"bpm\":" << status.bpm
         << ",\"confidence\":" << status.confidence
         << ",\"heart_rate_valid\":"
         << (status.heart_rate_valid ? "true" : "false")
         << ",\"heart_rate_invalid_reason\":\""
         << json_escape(status.heart_rate_invalid_reason)
         << "\",\"window_start_sec\":" << status.window_start_sec
         << ",\"window_end_sec\":" << status.window_end_sec
         << ",\"processing_exit_code\":" << status.processing_exit_code
         << ",\"output_directory\":\""
         << json_escape(status.output_directory)
         << "\",\"deep_enabled\":"
         << (status.deep_enabled ? "true" : "false")
         << ",\"deep_backend\":\"" << json_escape(status.deep_backend)
         << "\",\"deep_result_available\":"
         << (status.deep_result_available ? "true" : "false")
         << ",\"deep_bpm\":" << status.deep_bpm
         << ",\"deep_confidence\":" << status.deep_confidence
         << ",\"deep_inference_ms\":" << status.deep_inference_ms
         << ",\"deep_result_valid\":"
         << (status.deep_result_valid ? "true" : "false")
         << ",\"deep_invalid_reason\":\""
         << json_escape(status.deep_invalid_reason)
         << "\",\"measurement_available\":"
         << (status.measurement_available ? "true" : "false")
         << ",\"measurement\":" << measurement_json(status.measurement)
         << ",\"error_code\":\"" << json_escape(status.error_code)
         << "\",\"error_message\":\"" << json_escape(status.error_message)
         << "\"}";
  return output.str();
}

}  // namespace

std::string list_cameras_json() {
  const std::vector<CameraInfo> cameras =
      AndroidCameraSession::list_camera_infos();
  std::ostringstream output;
  output << "{\"cameras\":[";
  for (std::size_t index = 0; index < cameras.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << "{\"id\":\"" << json_escape(cameras[index].id)
           << "\",\"facing\":\"" << json_escape(cameras[index].facing)
           << "\",\"sensor_orientation\":" << cameras[index].sensor_orientation
           << "}";
  }
  output << "]}";
  return output.str();
}

std::int64_t create_camera_session(const std::string& camera_id, int width,
                                   int height, int fps) {
  auto session = std::make_shared<AndroidCameraSession>(
      CameraSessionConfig{camera_id, width, height, fps});
  const std::int64_t handle = next_handle.fetch_add(1);
  if (handle <= 0) {
    throw AppError(ErrorCode::NativeStateInvalid,
                   "camera session handle space exhausted");
  }
  std::lock_guard<std::mutex> lock(registry_mutex);
  registry.emplace(handle, std::move(session));
  return handle;
}

std::string configure_camera_processing(
    std::int64_t handle, const std::string& method,
    const std::string& cascade_path, const std::string& output_directory,
    bool deep_enabled, const std::string& model_path) {
  const auto session = lookup(handle);
  session->configure_processing(TraditionalProcessingConfig{
      method, cascade_path, output_directory, deep_enabled, model_path});
  return status_json(session->status());
}

void set_camera_preview_surface(std::int64_t handle, ::ANativeWindow* window) {
  const auto session = lookup(handle);
  session->set_preview_surface(window);
}

void set_camera_display_rotation(std::int64_t handle, int rotation_degrees) {
  const auto session = lookup(handle);
  session->set_display_rotation(rotation_degrees);
}

std::string start_camera_session(std::int64_t handle) {
  const auto session = lookup(handle);
  session->start();
  return status_json(session->status());
}

std::string stop_camera_session(std::int64_t handle) {
  const auto session = lookup(handle);
  session->stop();
  return status_json(session->status());
}

void destroy_camera_session(std::int64_t handle) noexcept {
  std::shared_ptr<AndroidCameraSession> session;
  {
    std::lock_guard<std::mutex> lock(registry_mutex);
    const auto found = registry.find(handle);
    if (found == registry.end()) {
      return;
    }
    session = std::move(found->second);
    registry.erase(found);
  }
  session->stop();
}

std::string camera_session_status_json(std::int64_t handle) {
  return status_json(lookup(handle)->status());
}

std::vector<std::uint8_t> camera_session_roi_jpeg(std::int64_t handle) {
  return lookup(handle)->latest_roi_jpeg();
}

std::string request_camera_color_diagnostic(std::int64_t handle) {
  const auto session = lookup(handle);
  session->request_color_diagnostic();
  return status_json(session->status());
}

std::string delete_camera_color_diagnostics(std::int64_t handle) {
  const auto session = lookup(handle);
  session->delete_color_diagnostics();
  return status_json(session->status());
}

}  // namespace rppg_qnn::android
