#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace rppg_qnn::android {

std::string list_cameras_json();
std::int64_t create_camera_session(const std::string& camera_id, int width,
                                   int height, int fps);
std::string configure_camera_processing(std::int64_t handle,
                                        const std::string& method,
                                        const std::string& cascade_path,
                                        const std::string& output_directory,
                                        bool deep_enabled,
                                        const std::string& model_path);
std::string start_camera_session(std::int64_t handle);
std::string stop_camera_session(std::int64_t handle);
void destroy_camera_session(std::int64_t handle) noexcept;
std::string camera_session_status_json(std::int64_t handle);
std::vector<std::uint8_t> camera_session_roi_jpeg(std::int64_t handle);

}  // namespace rppg_qnn::android
