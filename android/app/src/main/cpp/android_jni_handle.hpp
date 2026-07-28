#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ANativeWindow;

namespace rppg_qnn::android {

std::string list_cameras_json();
std::int64_t create_camera_session(const std::string& camera_id, int width,
                                   int height, int fps);
std::string configure_camera_processing(std::int64_t handle,
                                        const std::string& method,
                                        const std::string& cascade_path,
                                        const std::string& output_directory,
                                        const std::string& deep_model,
                                        const std::string& model_path);
void set_camera_preview_surface(std::int64_t handle, ::ANativeWindow* window);
void set_camera_display_rotation(std::int64_t handle, int rotation_degrees);
std::string start_camera_session(std::int64_t handle);
std::string stop_camera_session(std::int64_t handle);
void destroy_camera_session(std::int64_t handle) noexcept;
std::string camera_session_status_json(std::int64_t handle);
std::vector<std::uint8_t> camera_session_roi_jpeg(std::int64_t handle);
std::string request_camera_color_diagnostic(std::int64_t handle);
std::string delete_camera_color_diagnostics(std::int64_t handle);
std::string camera_session_waveform_metadata_json(std::int64_t handle,
                                                  bool deep);
std::vector<float> camera_session_waveform_values(std::int64_t handle,
                                                  bool deep);

}  // namespace rppg_qnn::android
