#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

struct ANativeWindow;

namespace rppg_qnn::android {

struct CameraSessionConfig {
  std::string camera_id;
  int width{640};
  int height{480};
  int fps{30};
};

struct TraditionalProcessingConfig {
  std::string method{"green"};
  std::string cascade_path;
  std::string output_directory;
  bool deep_enabled{false};
  std::string model_path;
};

struct CameraInfo {
  std::string id;
  std::string facing{"unknown"};
  int sensor_orientation{0};
};

struct CameraSessionStatus {
  std::string state{"created"};
  std::string camera_id;
  int requested_width{0};
  int requested_height{0};
  int requested_fps{0};
  int target_fps_min{0};
  int target_fps_max{0};
  int sensor_orientation{0};
  int display_rotation{0};
  int frame_rotation{0};
  double measured_fps{0.0};
  std::uint64_t accepted_frames{0};
  std::uint64_t dropped_frames{0};
  double last_timestamp_sec{0.0};
  bool processing_enabled{false};
  bool face_found{false};
  bool face_rect_available{false};
  double face_rect_x{0.0};
  double face_rect_y{0.0};
  double face_rect_w{0.0};
  double face_rect_h{0.0};
  bool preview_enabled{false};
  std::string traditional_method;
  bool heart_rate_available{false};
  double bpm{0.0};
  double confidence{0.0};
  bool heart_rate_valid{false};
  std::string heart_rate_invalid_reason;
  double window_start_sec{0.0};
  double window_end_sec{0.0};
  int processing_exit_code{0};
  std::string output_directory;
  bool deep_enabled{false};
  std::string deep_backend{"disabled"};
  bool deep_result_available{false};
  double deep_bpm{0.0};
  double deep_confidence{0.0};
  double deep_inference_ms{0.0};
  bool deep_result_valid{false};
  std::string deep_invalid_reason;
  std::string error_code;
  std::string error_message;
};

class AndroidCameraSession {
 public:
  explicit AndroidCameraSession(CameraSessionConfig config);
  ~AndroidCameraSession();

  AndroidCameraSession(const AndroidCameraSession&) = delete;
  AndroidCameraSession& operator=(const AndroidCameraSession&) = delete;

  static std::vector<std::string> list_cameras();
  static std::vector<CameraInfo> list_camera_infos();

  void configure_processing(TraditionalProcessingConfig config);
  void set_preview_surface(::ANativeWindow* window);
  void set_display_rotation(int rotation_degrees);
  void start();
  void stop() noexcept;
  [[nodiscard]] CameraSessionStatus status() const;
  [[nodiscard]] std::vector<std::uint8_t> latest_roi_jpeg() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rppg_qnn::android
