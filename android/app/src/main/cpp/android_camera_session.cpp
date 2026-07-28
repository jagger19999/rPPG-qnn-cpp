#include "android_camera_session.hpp"
#include "android_onnx_cpu_runtime.hpp"

#include "rppg_qnn/config.hpp"
#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/frame_source.hpp"
#include "rppg_qnn/latest_queue.hpp"
#include "rppg_qnn/pipeline.hpp"
#include "rppg_qnn/result_sink.hpp"
#include "rppg_qnn/roi_processor.hpp"
#include "rppg_qnn/yuv420.hpp"
#include "rppg_qnn/waveform_snapshot.hpp"

#include <android/log.h>
#include <android/native_window.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <camera/NdkCameraMetadataTags.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace rppg_qnn::android {
namespace {

constexpr char kLogTag[] = "rPPGCamera2";
constexpr int kMaxImages = 3;
constexpr std::size_t kFpsWindow = 120U;

void require_camera_ok(camera_status_t result, ErrorCode code,
                       const char* operation) {
  if (result != ACAMERA_OK) {
    throw AppError(code, std::string(operation) + " failed: " +
                             std::to_string(result));
  }
}

void require_media_ok(media_status_t result, ErrorCode code,
                      const char* operation) {
  if (result != AMEDIA_OK) {
    throw AppError(code, std::string(operation) + " failed: " +
                             std::to_string(result));
  }
}

std::string lens_facing_label(std::uint8_t facing) {
  switch (facing) {
    case ACAMERA_LENS_FACING_FRONT:
      return "front";
    case ACAMERA_LENS_FACING_BACK:
      return "back";
    case ACAMERA_LENS_FACING_EXTERNAL:
      return "external";
    default:
      return "unknown";
  }
}

std::string read_lens_facing(ACameraManager* manager,
                             const char* camera_id) {
  ACameraMetadata* characteristics = nullptr;
  const camera_status_t result = ACameraManager_getCameraCharacteristics(
      manager, camera_id, &characteristics);
  if (result != ACAMERA_OK || characteristics == nullptr) {
    return "unknown";
  }
  ACameraMetadata_const_entry entry{};
  if (ACameraMetadata_getConstEntry(characteristics, ACAMERA_LENS_FACING,
                                    &entry) != ACAMERA_OK ||
      entry.count == 0U || entry.data.u8 == nullptr) {
    ACameraMetadata_free(characteristics);
    return "unknown";
  }
  const std::string facing = lens_facing_label(entry.data.u8[0]);
  ACameraMetadata_free(characteristics);
  return facing;
}

int read_sensor_orientation(ACameraManager* manager, const char* camera_id) {
  ACameraMetadata* characteristics = nullptr;
  const camera_status_t result = ACameraManager_getCameraCharacteristics(
      manager, camera_id, &characteristics);
  if (result != ACAMERA_OK || characteristics == nullptr) {
    return 0;
  }
  ACameraMetadata_const_entry entry{};
  int orientation = 0;
  if (ACameraMetadata_getConstEntry(characteristics,
                                    ACAMERA_SENSOR_ORIENTATION, &entry) ==
          ACAMERA_OK &&
      entry.count > 0U && entry.data.i32 != nullptr) {
    orientation = entry.data.i32[0];
  }
  ACameraMetadata_free(characteristics);
  return orientation;
}

int normalize_rotation_degrees(int degrees) {
  int normalized = degrees % 360;
  if (normalized < 0) {
    normalized += 360;
  }
  return normalized;
}

int compute_frame_rotation_degrees(int sensor_orientation,
                                   int display_rotation_degrees,
                                   bool front_facing) {
  const int sensor = normalize_rotation_degrees(sensor_orientation);
  const int display = normalize_rotation_degrees(display_rotation_degrees);
  if (front_facing) {
    return normalize_rotation_degrees(sensor + display);
  }
  return normalize_rotation_degrees(sensor - display);
}

void orient_bgr_frame(const cv::Mat& source, int rotation_degrees,
                      cv::Mat* destination) {
  if (destination == nullptr) {
    return;
  }
  switch (normalize_rotation_degrees(rotation_degrees)) {
    case 90:
      cv::rotate(source, *destination, cv::ROTATE_90_CLOCKWISE);
      break;
    case 180:
      cv::rotate(source, *destination, cv::ROTATE_180);
      break;
    case 270:
      cv::rotate(source, *destination, cv::ROTATE_90_COUNTERCLOCKWISE);
      break;
    default:
      *destination = source;
      break;
  }
}

struct FpsRange {
  int min_fps{0};
  int max_fps{0};
};

FpsRange pick_target_fps_range(const ACameraMetadata_const_entry& entry,
                               int desired_fps) {
  if (entry.count < 2U || entry.data.i32 == nullptr ||
      entry.count % 2U != 0U) {
    return {desired_fps, desired_fps};
  }

  const std::size_t range_count = entry.count / 2U;
  FpsRange exact_match{0, 0};
  FpsRange includes_desired{0, 0};
  FpsRange closest{entry.data.i32[0], entry.data.i32[1]};
  int closest_distance =
      std::abs(closest.min_fps - desired_fps) +
      std::abs(closest.max_fps - desired_fps);

  for (std::size_t index = 0; index < range_count; ++index) {
    const int min_fps = entry.data.i32[index * 2U];
    const int max_fps = entry.data.i32[index * 2U + 1U];
    if (min_fps == desired_fps && max_fps == desired_fps) {
      exact_match = {min_fps, max_fps};
      break;
    }
    if (min_fps <= desired_fps && max_fps >= desired_fps) {
      if (includes_desired.min_fps == 0 && includes_desired.max_fps == 0) {
        includes_desired = {min_fps, max_fps};
      } else if (min_fps > includes_desired.min_fps) {
        includes_desired = {min_fps, max_fps};
      }
    }
    const int distance =
        std::abs(min_fps - desired_fps) + std::abs(max_fps - desired_fps);
    if (distance < closest_distance) {
      closest = {min_fps, max_fps};
      closest_distance = distance;
    }
  }

  if (exact_match.min_fps > 0) {
    return exact_match;
  }
  if (includes_desired.min_fps > 0) {
    return includes_desired;
  }
  return closest;
}

void apply_target_fps_range(ACameraManager* manager,
                            const std::string& camera_id,
                            ACaptureRequest* request, int desired_fps,
                            CameraSessionStatus* snapshot) {
  ACameraMetadata* characteristics = nullptr;
  require_camera_ok(
      ACameraManager_getCameraCharacteristics(manager, camera_id.c_str(),
                                              &characteristics),
      ErrorCode::CameraOpenFailed, "ACameraManager_getCameraCharacteristics");

  ACameraMetadata_const_entry entry{};
  const camera_status_t metadata_result = ACameraMetadata_getConstEntry(
      characteristics, ACAMERA_CONTROL_AE_AVAILABLE_TARGET_FPS_RANGES, &entry);
  FpsRange chosen{desired_fps, desired_fps};
  if (metadata_result == ACAMERA_OK) {
    chosen = pick_target_fps_range(entry, desired_fps);
  }

  const int range[] = {chosen.min_fps, chosen.max_fps};
  require_camera_ok(
      ACaptureRequest_setEntry_i32(request, ACAMERA_CONTROL_AE_TARGET_FPS_RANGE,
                                   2, range),
      ErrorCode::CameraOpenFailed, "ACaptureRequest_setEntry_i32 FPS range");
  ACameraMetadata_free(characteristics);

  if (snapshot != nullptr) {
    snapshot->target_fps_min = chosen.min_fps;
    snapshot->target_fps_max = chosen.max_fps;
  }
}

struct ImageDeleter {
  void operator()(AImage* image) const noexcept {
    if (image != nullptr) {
      AImage_delete(image);
    }
  }
};

using UniqueImage = std::unique_ptr<AImage, ImageDeleter>;

class LatestFrameQueue {
 public:
  bool push(FramePacket frame, bool* replaced) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (closed_) {
        return false;
      }
      *replaced = frame_.has_value();
      frame_ = std::move(frame);
    }
    condition_.notify_one();
    return true;
  }

  std::optional<FramePacket> wait_pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    condition_.wait(lock, [this] { return frame_.has_value() || closed_; });
    if (!frame_.has_value()) {
      return std::nullopt;
    }
    std::optional<FramePacket> result = std::move(frame_);
    frame_.reset();
    return result;
  }

  void close() noexcept {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
    }
    condition_.notify_all();
  }

  bool closed() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return closed_;
  }

 private:
  mutable std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<FramePacket> frame_;
  bool closed_{false};
};

class QueueFrameSource final : public FrameSource {
 public:
  QueueFrameSource(std::shared_ptr<LatestFrameQueue> queue, double nominal_fps)
      : queue_(std::move(queue)), nominal_fps_(nominal_fps) {}

  std::optional<FramePacket> read() override { return queue_->wait_pop(); }
  bool eof() const override { return queue_->closed(); }
  double nominal_fps() const override { return nominal_fps_; }

 private:
  std::shared_ptr<LatestFrameQueue> queue_;
  double nominal_fps_;
};

class StatusResultSink final : public IResultSink {
 public:
  using FrameCallback = std::function<void(const FrameHealth&)>;
  using HeartRateCallback = std::function<void(const HeartRateResult&)>;
  using MeasurementCallback = std::function<void(const MeasurementSnapshot&)>;
  using ErrorCallback =
      std::function<void(const std::string&, const std::string&)>;

  StatusResultSink(const std::filesystem::path& output_directory,
                   FrameCallback frame_callback,
                   HeartRateCallback heart_rate_callback,
                   MeasurementCallback measurement_callback,
                   ErrorCallback error_callback)
      : sink_(output_directory),
        frame_callback_(std::move(frame_callback)),
        heart_rate_callback_(std::move(heart_rate_callback)),
        measurement_callback_(std::move(measurement_callback)),
        error_callback_(std::move(error_callback)) {}

  void publish(const PreflightResult& result) override {
    sink_.publish(result);
  }

  void publish_runtime_error(const std::string& error_code,
                             const std::string& message) override {
    sink_.publish_runtime_error(error_code, message);
    error_callback_(error_code, message);
  }

  void publish(const FrameHealth& result) override {
    sink_.publish(result);
    frame_callback_(result);
  }

  void publish(const HeartRateResult& result) override {
    sink_.publish(result);
    heart_rate_callback_(result);
  }

  void publish(const MeasurementSnapshot& result) override {
    sink_.publish(result);
    measurement_callback_(result);
  }

  void close(int exit_code) override { sink_.close(exit_code); }

 private:
  ResultSink sink_;
  FrameCallback frame_callback_;
  HeartRateCallback heart_rate_callback_;
  MeasurementCallback measurement_callback_;
  ErrorCallback error_callback_;
};

class ThumbnailRoi final : public IRoiProcessor {
 public:
  using PublishCallback =
      std::function<void(const RoiPacket&, int frame_width, int frame_height)>;

  ThumbnailRoi(std::unique_ptr<IRoiProcessor> inner, PublishCallback publish);

  RoiPacket process(const FramePacket& frame) override;

 private:
  std::unique_ptr<IRoiProcessor> inner_;
  PublishCallback publish_;
};

struct ColorDiagnosticJob {
  std::uint64_t frame_id{0};
  int width{0};
  int height{0};
  int rotation_degrees{0};
  std::array<int, 3> row_strides{};
  std::array<int, 3> pixel_strides{};
  std::array<std::vector<std::uint8_t>, 3> planes;
  std::vector<std::uint8_t> roi_jpeg;
  std::filesystem::path root;
};

void write_bytes(const std::filesystem::path& path,
                 const std::vector<std::uint8_t>& values) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("could not open " + path.string());
  output.write(reinterpret_cast<const char*>(values.data()),
               static_cast<std::streamsize>(values.size()));
  if (!output) throw std::runtime_error("could not write " + path.string());
}

std::pair<double, double> byte_mean_std(
    const std::vector<std::uint8_t>& values) {
  if (values.empty()) return {0.0, 0.0};
  double mean = 0.0;
  for (std::uint8_t value : values) mean += value;
  mean /= static_cast<double>(values.size());
  double variance = 0.0;
  for (std::uint8_t value : values) {
    const double centered = static_cast<double>(value) - mean;
    variance += centered * centered;
  }
  return {mean, std::sqrt(variance / static_cast<double>(values.size()))};
}

}  // namespace

struct AndroidCameraSession::Impl {
  explicit Impl(CameraSessionConfig requested_config)
      : config(std::move(requested_config)) {
    snapshot.camera_id = config.camera_id;
    snapshot.requested_width = config.width;
    snapshot.requested_height = config.height;
    snapshot.requested_fps = config.fps;
  }

  ~Impl() {
    stop();
    if (preview_window != nullptr) {
      ANativeWindow_release(preview_window);
      preview_window = nullptr;
    }
  }

  void set_error(ErrorCode code, std::string message) {
    __android_log_print(ANDROID_LOG_ERROR, kLogTag, "%.*s: %s",
                        static_cast<int>(to_string(code).size()),
                        to_string(code).data(), message.c_str());
    std::lock_guard<std::mutex> lock(status_mutex);
    snapshot.state = "error";
    snapshot.error_code = std::string(to_string(code));
    snapshot.error_message = std::move(message);
  }

  void configure_processing(TraditionalProcessingConfig requested_config) {
    if (requested_config.method != "green" &&
        requested_config.method != "pos" &&
        requested_config.method != "chrom") {
      throw AppError(ErrorCode::ConfigInvalid,
                     "traditional method must be green, pos, or chrom");
    }
    if (requested_config.cascade_path.empty() ||
        requested_config.output_directory.empty()) {
      throw AppError(ErrorCode::ConfigInvalid,
                     "cascade path and output directory must not be empty");
    }
    if (requested_config.deep_model != DeepModel::Disabled &&
        (requested_config.model_path.empty() ||
         !std::filesystem::is_regular_file(requested_config.model_path))) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     std::string(to_string(requested_config.deep_model)) +
                         " ONNX model is missing from app storage");
    }
    {
      RoiProcessor cascade_validator(requested_config.cascade_path);
      (void)cascade_validator;
    }
    std::error_code output_error;
    std::filesystem::create_directories(requested_config.output_directory,
                                        output_error);
    if (output_error) {
      throw AppError(ErrorCode::OutputWriteFailed,
                     "could not create Android output directory: " +
                         output_error.message());
    }
    std::lock_guard<std::mutex> lock(status_mutex);
    if (snapshot.state == "running" || snapshot.state == "starting" ||
        snapshot.state == "stopping") {
      throw AppError(ErrorCode::NativeStateInvalid,
                     "processing cannot be configured while camera is active");
    }
    processing_config = std::move(requested_config);
    snapshot.processing_enabled = true;
    snapshot.traditional_method = processing_config->method;
    snapshot.output_directory = processing_config->output_directory;
    snapshot.deep_model = std::string(to_string(processing_config->deep_model));
    snapshot.deep_enabled =
        processing_config->deep_model != DeepModel::Disabled;
    snapshot.deep_backend =
        snapshot.deep_enabled ? "ONNX_RUNTIME_CPU" : "disabled";
  }

  void start_processing() {
    if (!processing_config.has_value()) {
      return;
    }

    const TraditionalProcessingConfig requested = *processing_config;
    frame_queue = std::make_shared<LatestFrameQueue>();
    processing_stop.store(false);
    color_diagnostic_queue =
        std::make_unique<LatestQueue<ColorDiagnosticJob>>();
    color_diagnostic_thread = std::thread([this] { run_color_diagnostics(); });

    AppConfig pipeline_config;
    pipeline_config.camera = config.camera_id;
    pipeline_config.width = config.width;
    pipeline_config.height = config.height;
    pipeline_config.fps = static_cast<double>(config.fps);
    pipeline_config.traditional = requested.method;
    pipeline_config.deep = std::string(to_string(requested.deep_model));
    pipeline_config.output = requested.output_directory;

    PipelineDependencies dependencies;
    const std::shared_ptr<LatestFrameQueue> queue = frame_queue;
    dependencies.make_source = [queue, fps = pipeline_config.fps] {
      return std::make_unique<QueueFrameSource>(queue, fps);
    };
    dependencies.make_roi = [this, cascade = requested.cascade_path] {
      ThumbnailRoi::PublishCallback publish =
          [this](const RoiPacket& packet, int frame_width, int frame_height) {
            update_face_rect(packet, frame_width, frame_height);
            maybe_publish_roi_jpeg(packet);
          };
      return std::make_unique<ThumbnailRoi>(
          std::make_unique<RoiProcessor>(cascade), std::move(publish));
    };
    switch (requested.deep_model) {
      case DeepModel::Disabled:
        break;
      case DeepModel::Tscan:
      case DeepModel::EfficientPhys:
        // Construct synchronously before Camera2 starts accepting frames. This
        // makes ORT's name/type/shape inspection a true start-time gate.
        {
          auto runtime_holder =
              std::make_shared<std::unique_ptr<IDeepRuntime>>(
                  make_onnx_cpu_runtime(requested.deep_model,
                                        requested.model_path));
          dependencies.make_deep_runtime = [runtime_holder] {
            return std::move(*runtime_holder);
          };
        }
        break;
    }
    dependencies.make_sink = [this](
                                 const std::filesystem::path& output_directory) {
      return std::make_unique<StatusResultSink>(
          output_directory,
          [this](const FrameHealth& health) {
            std::lock_guard<std::mutex> lock(status_mutex);
            snapshot.face_found = health.face_found;
            snapshot.performance = health.performance;
          },
          [this](const HeartRateResult& result) {
            std::lock_guard<std::mutex> lock(status_mutex);
            const double sample_rate = waveform_sample_rate_hz(result);
            WaveformSnapshot waveform = make_waveform_snapshot(
                result.waveform, result.method, sample_rate, result.is_valid,
                result.invalid_reason);
            if (result.method == "TSCAN" ||
                result.method == "EFFICIENTPHYS" || result.method == "DEEP") {
              snapshot.deep_result_available = true;
              snapshot.deep_bpm = result.bpm;
              snapshot.deep_raw_bpm = result.raw_bpm;
              snapshot.deep_display_bpm = result.display_bpm;
              snapshot.deep_confidence = result.confidence;
              snapshot.deep_window_materialization_ms =
                  result.window_materialization_ms;
              snapshot.deep_preprocess_ms = result.preprocess_ms;
              snapshot.deep_runtime_ms = result.runtime_ms;
              snapshot.deep_postprocess_ms = result.postprocess_ms;
              snapshot.deep_inference_ms = result.inference_ms;
              snapshot.deep_result_valid = result.is_valid;
              snapshot.deep_stability_valid = result.stability_valid;
              snapshot.deep_correction_reason = result.correction_reason;
              snapshot.deep_invalid_reason = result.invalid_reason;
              if (waveform.available) {
                waveform.revision = ++deep_waveform_revision_;
                deep_waveform_ = std::move(waveform);
                snapshot.deep_waveform_revision = deep_waveform_revision_;
              }
            } else {
              snapshot.heart_rate_available = true;
              snapshot.bpm = result.bpm;
              snapshot.confidence = result.confidence;
              snapshot.heart_rate_valid = result.is_valid;
              snapshot.heart_rate_invalid_reason = result.invalid_reason;
              snapshot.window_start_sec = result.window_start_sec;
              snapshot.window_end_sec = result.window_end_sec;
              if (waveform.available) {
                waveform.revision = ++traditional_waveform_revision_;
                traditional_waveform_ = std::move(waveform);
                snapshot.traditional_waveform_revision =
                    traditional_waveform_revision_;
              }
            }
          },
          [this](const MeasurementSnapshot& result) {
            std::lock_guard<std::mutex> lock(status_mutex);
            snapshot.measurement_available = true;
            snapshot.measurement = result;
          },
          [this](const std::string& code, const std::string& message) {
            std::lock_guard<std::mutex> lock(status_mutex);
            snapshot.state = "error";
            snapshot.error_code = code;
            snapshot.error_message = message;
          });
    };
    dependencies.should_stop = [this] { return processing_stop.load(); };

    processing_thread = std::thread(
        [this, pipeline_config = std::move(pipeline_config),
         dependencies = std::move(dependencies)]() mutable {
          const int exit_code =
              Pipeline(std::move(pipeline_config), std::move(dependencies))
                  .run();
          std::lock_guard<std::mutex> lock(status_mutex);
          snapshot.processing_exit_code = exit_code;
          if (exit_code != 0 && snapshot.error_code.empty()) {
            snapshot.state = "error";
            snapshot.error_code = "PROCESSING_FAILED";
            snapshot.error_message =
                "traditional processing exited with code " +
                std::to_string(exit_code);
          }
        });
  }

  void stop_processing() noexcept {
    processing_stop.store(true);
    if (frame_queue != nullptr) {
      frame_queue->close();
    }
    if (processing_thread.joinable()) {
      processing_thread.join();
    }
    if (color_diagnostic_queue != nullptr) {
      color_diagnostic_queue->close();
    }
    if (color_diagnostic_thread.joinable()) {
      color_diagnostic_thread.join();
    }
    color_diagnostic_queue.reset();
    frame_queue.reset();
    clear_roi_jpeg();
  }

  void request_color_diagnostic() {
    std::lock_guard<std::mutex> lock(status_mutex);
    if (snapshot.state != "running" || !processing_config.has_value()) {
      throw AppError(ErrorCode::NativeStateInvalid,
                     "color diagnostic requires a running processed camera");
    }
    color_diagnostic_requested.store(true);
    snapshot.color_diagnostic_state = "pending";
    snapshot.color_diagnostic_path.clear();
    snapshot.color_diagnostic_finding.clear();
  }

  void delete_color_diagnostics() {
    std::filesystem::path diagnostics;
    {
      std::lock_guard<std::mutex> lock(status_mutex);
      if (!processing_config.has_value()) return;
      diagnostics = std::filesystem::path(processing_config->output_directory) /
                    "color_diagnostics";
    }
    std::error_code error;
    std::filesystem::remove_all(diagnostics, error);
    std::lock_guard<std::mutex> lock(status_mutex);
    snapshot.color_diagnostic_state = error ? "delete_failed" : "idle";
    snapshot.color_diagnostic_path.clear();
    snapshot.color_diagnostic_finding =
        error ? error.message() : "diagnostics_deleted";
  }

  void run_color_diagnostics() noexcept {
    while (color_diagnostic_queue != nullptr) {
      auto job = color_diagnostic_queue->wait_pop(std::chrono::hours(24));
      if (!job.has_value()) {
        if (color_diagnostic_queue->closed()) return;
        continue;
      }
      try {
        const std::filesystem::path base =
            job->root / "color_diagnostics" /
            ("frame-" + std::to_string(job->frame_id));
        std::filesystem::create_directories(base);
        for (std::size_t index = 0; index < job->planes.size(); ++index) {
          write_bytes(base / (std::string(1, "yuv"[index]) + "_plane.bin"),
                      job->planes[index]);
        }
        if (!job->roi_jpeg.empty()) {
          write_bytes(base / "roi.jpg", job->roi_jpeg);
        }
        const Yuv420View view{
            job->width,
            job->height,
            {job->planes[0].data(), job->planes[0].size(),
             job->row_strides[0], job->pixel_strides[0]},
            {job->planes[1].data(), job->planes[1].size(),
             job->row_strides[1], job->pixel_strides[1]},
            {job->planes[2].data(), job->planes[2].size(),
             job->row_strides[2], job->pixel_strides[2]}};
        struct Variant {
          const char* name;
          YuvColorSpec spec;
        };
        const std::array<Variant, 8> variants{{
            {"bt601_limited_uv", {YuvMatrix::Bt601, YuvRange::Limited,
                                   ChromaOrder::Uv}},
            {"bt601_limited_vu", {YuvMatrix::Bt601, YuvRange::Limited,
                                   ChromaOrder::Vu}},
            {"bt709_limited_uv", {YuvMatrix::Bt709, YuvRange::Limited,
                                   ChromaOrder::Uv}},
            {"bt709_limited_vu", {YuvMatrix::Bt709, YuvRange::Limited,
                                   ChromaOrder::Vu}},
            {"bt601_full_uv", {YuvMatrix::Bt601, YuvRange::Full,
                                ChromaOrder::Uv}},
            {"bt601_full_vu", {YuvMatrix::Bt601, YuvRange::Full,
                                ChromaOrder::Vu}},
            {"bt709_full_uv", {YuvMatrix::Bt709, YuvRange::Full,
                                ChromaOrder::Uv}},
            {"bt709_full_vu", {YuvMatrix::Bt709, YuvRange::Full,
                                ChromaOrder::Vu}},
        }};
        std::ostringstream manifest;
        manifest << std::setprecision(8)
                 << "{\"schema_version\":1,\"frame_id\":" << job->frame_id
                 << ",\"width\":" << job->width << ",\"height\":"
                 << job->height << ",\"rotation_degrees\":"
                 << job->rotation_degrees << ",\"planes\":[";
        for (std::size_t index = 0; index < job->planes.size(); ++index) {
          if (index != 0U) manifest << ',';
          const auto [mean, standard_deviation] =
              byte_mean_std(job->planes[index]);
          manifest << "{\"index\":" << index << ",\"size\":"
                   << job->planes[index].size() << ",\"row_stride\":"
                   << job->row_strides[index] << ",\"pixel_stride\":"
                   << job->pixel_strides[index] << ",\"mean\":" << mean
                   << ",\"std\":" << standard_deviation << '}';
        }
        manifest << "],\"variants\":[";
        for (std::size_t index = 0; index < variants.size(); ++index) {
          if (index != 0U) manifest << ',';
          std::vector<std::uint8_t> converted =
              yuv420_to_bgr(view, variants[index].spec);
          cv::Mat wrapped(job->height, job->width, CV_8UC3, converted.data());
          cv::Mat oriented;
          orient_bgr_frame(wrapped, job->rotation_degrees, &oriented);
          const cv::Scalar mean = cv::mean(oriented);
          const std::string filename =
              std::string(variants[index].name) + ".png";
          if (!cv::imwrite((base / filename).string(), oriented)) {
            throw std::runtime_error("could not encode diagnostic PNG");
          }
          manifest << "{\"name\":\"" << variants[index].name
                   << "\",\"b_mean\":" << mean[0]
                   << ",\"g_mean\":" << mean[1]
                   << ",\"r_mean\":" << mean[2] << '}';
        }
        manifest << "],\"finding\":\"INSUFFICIENT_EVIDENCE\","
                    "\"note\":\"Compare the eight PNG variants with the camera preview; no automatic correction was applied.\"}";
        std::ofstream manifest_file(base / "manifest.json",
                                    std::ios::out | std::ios::trunc);
        manifest_file << manifest.str();
        if (!manifest_file) {
          throw std::runtime_error("could not write diagnostic manifest");
        }

        std::vector<std::filesystem::path> packages;
        for (const auto& entry :
             std::filesystem::directory_iterator(base.parent_path())) {
          if (entry.is_directory()) packages.push_back(entry.path());
        }
        std::sort(packages.begin(), packages.end());
        while (packages.size() > 10U) {
          std::error_code remove_error;
          std::filesystem::remove_all(packages.front(), remove_error);
          packages.erase(packages.begin());
        }
        std::lock_guard<std::mutex> lock(status_mutex);
        snapshot.color_diagnostic_state = "complete";
        snapshot.color_diagnostic_path = base.string();
        snapshot.color_diagnostic_finding = "INSUFFICIENT_EVIDENCE";
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(status_mutex);
        snapshot.color_diagnostic_state = "failed";
        snapshot.color_diagnostic_finding = error.what();
      }
    }
  }

  void clear_roi_jpeg() {
    std::lock_guard<std::mutex> lock(status_mutex);
    latest_roi_jpeg_.clear();
    last_roi_jpeg_sec_ = 0.0;
  }

  void update_face_rect(const RoiPacket& packet, int frame_width,
                        int frame_height) {
    std::lock_guard<std::mutex> lock(status_mutex);
    if (packet.face.has_value() && frame_width > 0 && frame_height > 0) {
      const FaceBox& face = *packet.face;
      snapshot.face_rect_x =
          static_cast<double>(face.x) / static_cast<double>(frame_width);
      snapshot.face_rect_y =
          static_cast<double>(face.y) / static_cast<double>(frame_height);
      snapshot.face_rect_w =
          static_cast<double>(face.width) / static_cast<double>(frame_width);
      snapshot.face_rect_h =
          static_cast<double>(face.height) / static_cast<double>(frame_height);
      snapshot.face_rect_available = true;
      snapshot.face_found = true;
      return;
    }
    snapshot.face_rect_available = false;
    snapshot.face_found = false;
  }

  void set_preview_surface(::ANativeWindow* window) {
    std::lock_guard<std::mutex> lock(status_mutex);
    if (snapshot.state == "running" || snapshot.state == "starting" ||
        snapshot.state == "stopping") {
      throw AppError(ErrorCode::NativeStateInvalid,
                     "preview surface must be set before camera start");
    }
    if (preview_window != nullptr) {
      ANativeWindow_release(preview_window);
      preview_window = nullptr;
    }
    preview_window = window;
    snapshot.preview_enabled = preview_window != nullptr;
  }

  void set_display_rotation(int rotation_degrees) {
    std::lock_guard<std::mutex> lock(status_mutex);
    if (snapshot.state == "running" || snapshot.state == "starting" ||
        snapshot.state == "stopping") {
      throw AppError(ErrorCode::NativeStateInvalid,
                     "display rotation must be set before camera start");
    }
    display_rotation_degrees_ =
        normalize_rotation_degrees(rotation_degrees);
    snapshot.display_rotation = display_rotation_degrees_;
  }

  void maybe_publish_roi_jpeg(const RoiPacket& packet) {
    if (packet.roi_bgr.empty() || !packet.face.has_value()) {
      std::lock_guard<std::mutex> lock(status_mutex);
      latest_roi_jpeg_.clear();
      return;
    }
    {
      std::lock_guard<std::mutex> lock(status_mutex);
      if (packet.timestamp_sec - last_roi_jpeg_sec_ < 0.25) {
        return;
      }
    }
    cv::Mat small;
    cv::resize(packet.roi_bgr, small, cv::Size(160, 160));
    std::vector<std::uint8_t> encoded;
    if (!cv::imencode(".jpg", small, encoded,
                       {cv::IMWRITE_JPEG_QUALITY, 80})) {
      std::lock_guard<std::mutex> lock(status_mutex);
      latest_roi_jpeg_.clear();
      return;
    }
    std::lock_guard<std::mutex> lock(status_mutex);
    latest_roi_jpeg_ = std::move(encoded);
    last_roi_jpeg_sec_ = packet.timestamp_sec;
  }

  std::vector<std::uint8_t> latest_roi_jpeg() const {
    std::lock_guard<std::mutex> lock(status_mutex);
    return latest_roi_jpeg_;
  }

  WaveformSnapshot latest_waveform(bool deep) const {
    std::lock_guard<std::mutex> lock(status_mutex);
    return deep ? deep_waveform_ : traditional_waveform_;
  }

  void start() {
    if (config.camera_id.empty()) {
      throw AppError(ErrorCode::CameraIdUnavailable,
                     "camera ID must not be empty");
    }
    if (config.width <= 0 || config.height <= 0 || config.width % 2 != 0 ||
        config.height % 2 != 0 || config.fps <= 0) {
      throw AppError(ErrorCode::ConfigInvalid,
                     "camera width/height must be positive and even and FPS "
                     "must be positive");
    }
    {
      std::lock_guard<std::mutex> lock(status_mutex);
      if (snapshot.state == "running" || snapshot.state == "starting") {
        return;
      }
      snapshot.state = "starting";
      snapshot.error_code.clear();
      snapshot.error_message.clear();
      snapshot.measured_fps = 0.0;
      snapshot.accepted_frames = 0;
      snapshot.dropped_frames = 0;
      snapshot.last_timestamp_sec = 0.0;
      snapshot.face_found = false;
      snapshot.face_rect_available = false;
      snapshot.face_rect_x = 0.0;
      snapshot.face_rect_y = 0.0;
      snapshot.face_rect_w = 0.0;
      snapshot.face_rect_h = 0.0;
      snapshot.preview_enabled = preview_window != nullptr;
      snapshot.heart_rate_available = false;
      snapshot.bpm = 0.0;
      snapshot.confidence = 0.0;
      snapshot.heart_rate_valid = false;
      snapshot.heart_rate_invalid_reason.clear();
      snapshot.window_start_sec = 0.0;
      snapshot.window_end_sec = 0.0;
      snapshot.processing_exit_code = 0;
      snapshot.deep_result_available = false;
      snapshot.deep_bpm = 0.0;
      snapshot.deep_confidence = 0.0;
      snapshot.deep_raw_bpm = 0.0;
      snapshot.deep_display_bpm = 0.0;
      snapshot.deep_window_materialization_ms = 0.0;
      snapshot.deep_preprocess_ms = 0.0;
      snapshot.deep_runtime_ms = 0.0;
      snapshot.deep_postprocess_ms = 0.0;
      snapshot.deep_inference_ms = 0.0;
      snapshot.deep_result_valid = false;
      snapshot.deep_stability_valid = false;
      snapshot.deep_correction_reason.clear();
      snapshot.deep_invalid_reason.clear();
      snapshot.measurement_available = false;
      snapshot.measurement = MeasurementSnapshot{};
      traditional_waveform_ = {};
      deep_waveform_ = {};
      snapshot.traditional_waveform_revision = 0;
      snapshot.deep_waveform_revision = 0;
      snapshot.deep_frames_collected = 0;
      traditional_waveform_revision_ = 0;
      deep_waveform_revision_ = 0;
      timestamps.clear();
    }
    clear_roi_jpeg();

    try {
      start_processing();
      manager = ACameraManager_create();
      if (manager == nullptr) {
        throw AppError(ErrorCode::CameraOpenFailed,
                       "ACameraManager_create returned null");
      }

      require_media_ok(
          AImageReader_new(config.width, config.height,
                           AIMAGE_FORMAT_YUV_420_888, kMaxImages, &reader),
          ErrorCode::CameraImageInvalid, "AImageReader_new");
      require_media_ok(AImageReader_getWindow(reader, &window),
                       ErrorCode::CameraImageInvalid,
                       "AImageReader_getWindow");

      AImageReader_ImageListener image_listener{this, &Impl::on_image_available};
      {
        std::lock_guard<std::mutex> callback_lock(callback_mutex);
        callbacks_enabled = true;
      }
      require_media_ok(AImageReader_setImageListener(reader, &image_listener),
                       ErrorCode::CameraImageInvalid,
                       "AImageReader_setImageListener");

      ACameraDevice_StateCallbacks device_callbacks{
          this, &Impl::on_device_disconnected, &Impl::on_device_error};
      const camera_status_t open_result = ACameraManager_openCamera(
          manager, config.camera_id.c_str(), &device_callbacks, &device);
      if (open_result == ACAMERA_ERROR_PERMISSION_DENIED) {
        throw AppError(ErrorCode::CameraPermissionDenied,
                       "camera permission denied");
      }
      require_camera_ok(open_result, ErrorCode::CameraOpenFailed,
                        "ACameraManager_openCamera");

      sensor_orientation_ =
          read_sensor_orientation(manager, config.camera_id.c_str());
      const bool front_facing =
          read_lens_facing(manager, config.camera_id.c_str()) == "front";
      frame_rotation_degrees_ = compute_frame_rotation_degrees(
          sensor_orientation_, display_rotation_degrees_, front_facing);
      {
        std::lock_guard<std::mutex> lock(status_mutex);
        snapshot.sensor_orientation = sensor_orientation_;
        snapshot.display_rotation = display_rotation_degrees_;
        snapshot.frame_rotation = frame_rotation_degrees_;
      }

      require_camera_ok(
          ACaptureSessionOutputContainer_create(&output_container),
          ErrorCode::CameraOpenFailed,
          "ACaptureSessionOutputContainer_create");
      require_camera_ok(ACaptureSessionOutput_create(window, &session_output),
                        ErrorCode::CameraOpenFailed,
                        "ACaptureSessionOutput_create");
      require_camera_ok(
          ACaptureSessionOutputContainer_add(output_container, session_output),
          ErrorCode::CameraOpenFailed,
          "ACaptureSessionOutputContainer_add");
      require_camera_ok(ACameraOutputTarget_create(window, &output_target),
                        ErrorCode::CameraOpenFailed,
                        "ACameraOutputTarget_create");
      require_camera_ok(
          ACameraDevice_createCaptureRequest(device, TEMPLATE_PREVIEW, &request),
          ErrorCode::CameraOpenFailed,
          "ACameraDevice_createCaptureRequest");
      require_camera_ok(ACaptureRequest_addTarget(request, output_target),
                        ErrorCode::CameraOpenFailed,
                        "ACaptureRequest_addTarget");

      const bool use_preview = preview_window != nullptr;
      if (use_preview) {
        require_camera_ok(
            ACaptureSessionOutput_create(preview_window, &preview_session_output),
            ErrorCode::CameraOpenFailed, "ACaptureSessionOutput_create preview");
        require_camera_ok(
            ACaptureSessionOutputContainer_add(output_container,
                                               preview_session_output),
            ErrorCode::CameraOpenFailed,
            "ACaptureSessionOutputContainer_add preview");
        require_camera_ok(
            ACameraOutputTarget_create(preview_window, &preview_output_target),
            ErrorCode::CameraOpenFailed, "ACameraOutputTarget_create preview");
        require_camera_ok(
            ACaptureRequest_addTarget(request, preview_output_target),
            ErrorCode::CameraOpenFailed,
            "ACaptureRequest_addTarget preview");
      }

      {
        std::lock_guard<std::mutex> lock(status_mutex);
        apply_target_fps_range(manager, config.camera_id, request, config.fps,
                               &snapshot);
      }

      ACameraCaptureSession_stateCallbacks session_callbacks{
          this, &Impl::on_session_closed, &Impl::on_session_ready,
          &Impl::on_session_active};
      camera_status_t session_result = ACameraDevice_createCaptureSession(
          device, output_container, &session_callbacks, &capture_session);
      if (session_result != ACAMERA_OK && use_preview) {
        __android_log_print(
            ANDROID_LOG_WARN, kLogTag,
            "dual output capture session failed (%d); falling back to "
            "ImageReader only",
            session_result);
        if (preview_output_target != nullptr) {
          (void)ACaptureRequest_removeTarget(request, preview_output_target);
          ACameraOutputTarget_free(preview_output_target);
          preview_output_target = nullptr;
        }
        if (output_container != nullptr && preview_session_output != nullptr) {
          (void)ACaptureSessionOutputContainer_remove(output_container,
                                                      preview_session_output);
        }
        if (preview_session_output != nullptr) {
          ACaptureSessionOutput_free(preview_session_output);
          preview_session_output = nullptr;
        }
        session_result = ACameraDevice_createCaptureSession(
            device, output_container, &session_callbacks, &capture_session);
        {
          std::lock_guard<std::mutex> lock(status_mutex);
          snapshot.preview_enabled = false;
          if (snapshot.error_message.empty()) {
            snapshot.error_message =
                "preview_unavailable: dual output session rejected";
          }
        }
      }
      require_camera_ok(session_result, ErrorCode::CameraOpenFailed,
                        "ACameraDevice_createCaptureSession");

      ACaptureRequest* requests[] = {request};
      require_camera_ok(ACameraCaptureSession_setRepeatingRequest(
                            capture_session, nullptr, 1, requests, nullptr),
                        ErrorCode::CameraOpenFailed,
                        "ACameraCaptureSession_setRepeatingRequest");
      std::lock_guard<std::mutex> lock(status_mutex);
      if (snapshot.state == "starting") {
        snapshot.state = "running";
      }
      if (use_preview && preview_session_output != nullptr) {
        snapshot.preview_enabled = true;
      }
    } catch (const AppError& error) {
      const ErrorCode code = error.code();
      const std::string message = error.what();
      stop_resources();
      stop_processing();
      set_error(code, message);
      throw;
    } catch (const std::exception& error) {
      const std::string message = error.what();
      stop_resources();
      stop_processing();
      set_error(ErrorCode::CameraOpenFailed, message);
      throw AppError(ErrorCode::CameraOpenFailed, message);
    }
  }

  void stop() noexcept {
    {
      std::lock_guard<std::mutex> lock(status_mutex);
      if (snapshot.state == "stopped" || snapshot.state == "created") {
        snapshot.state = "stopped";
        return;
      }
      if (snapshot.state != "error") {
        snapshot.state = "stopping";
      }
    }
    stop_resources();
    stop_processing();
    std::lock_guard<std::mutex> lock(status_mutex);
    if (snapshot.state != "error") {
      snapshot.state = "stopped";
    }
  }

  void stop_resources() noexcept {
    {
      std::lock_guard<std::mutex> lock(callback_mutex);
      callbacks_enabled = false;
    }
    if (reader != nullptr) {
      (void)AImageReader_setImageListener(reader, nullptr);
    }
    {
      std::unique_lock<std::mutex> lock(callback_mutex);
      callback_condition.wait(lock, [this] { return active_callbacks == 0U; });
    }

    if (capture_session != nullptr) {
      (void)ACameraCaptureSession_stopRepeating(capture_session);
      (void)ACameraCaptureSession_abortCaptures(capture_session);
      ACameraCaptureSession_close(capture_session);
      capture_session = nullptr;
    }
    if (request != nullptr) {
      ACaptureRequest_free(request);
      request = nullptr;
    }
    if (output_target != nullptr) {
      ACameraOutputTarget_free(output_target);
      output_target = nullptr;
    }
    if (preview_output_target != nullptr) {
      ACameraOutputTarget_free(preview_output_target);
      preview_output_target = nullptr;
    }
    if (output_container != nullptr && session_output != nullptr) {
      (void)ACaptureSessionOutputContainer_remove(output_container,
                                                  session_output);
    }
    if (session_output != nullptr) {
      ACaptureSessionOutput_free(session_output);
      session_output = nullptr;
    }
    if (output_container != nullptr && preview_session_output != nullptr) {
      (void)ACaptureSessionOutputContainer_remove(output_container,
                                                  preview_session_output);
    }
    if (preview_session_output != nullptr) {
      ACaptureSessionOutput_free(preview_session_output);
      preview_session_output = nullptr;
    }
    if (output_container != nullptr) {
      ACaptureSessionOutputContainer_free(output_container);
      output_container = nullptr;
    }
    if (device != nullptr) {
      (void)ACameraDevice_close(device);
      device = nullptr;
    }
    if (reader != nullptr) {
      AImageReader_delete(reader);
      reader = nullptr;
      window = nullptr;
    }
    if (manager != nullptr) {
      ACameraManager_delete(manager);
      manager = nullptr;
    }
  }

  CameraSessionStatus status() const {
    std::lock_guard<std::mutex> lock(status_mutex);
    return snapshot;
  }

  bool begin_callback() {
    std::lock_guard<std::mutex> lock(callback_mutex);
    if (!callbacks_enabled) {
      return false;
    }
    ++active_callbacks;
    return true;
  }

  void end_callback() noexcept {
    std::lock_guard<std::mutex> lock(callback_mutex);
    --active_callbacks;
    callback_condition.notify_all();
  }

  static void on_image_available(void* context, AImageReader* image_reader) {
    auto* self = static_cast<Impl*>(context);
    if (self == nullptr || !self->begin_callback()) {
      return;
    }
    struct CallbackGuard {
      Impl* impl;
      ~CallbackGuard() { impl->end_callback(); }
    } callback_guard{self};

    AImage* raw_image = nullptr;
    const media_status_t acquire_result =
        AImageReader_acquireLatestImage(image_reader, &raw_image);
    UniqueImage image(raw_image);
    if (acquire_result == AMEDIA_IMGREADER_NO_BUFFER_AVAILABLE) {
      std::lock_guard<std::mutex> lock(self->status_mutex);
      ++self->snapshot.dropped_frames;
      return;
    }
    if (acquire_result != AMEDIA_OK || image == nullptr) {
      self->set_error(ErrorCode::CameraImageInvalid,
                      "AImageReader_acquireLatestImage failed: " +
                          std::to_string(acquire_result));
      return;
    }

    try {
      self->process_image(image.get());
    } catch (const std::exception& error) {
      self->set_error(ErrorCode::CameraImageInvalid, error.what());
    }
  }

  void process_image(AImage* image) {
    int32_t width = 0;
    int32_t height = 0;
    int32_t format = 0;
    int64_t timestamp_ns = 0;
    require_media_ok(AImage_getWidth(image, &width),
                     ErrorCode::CameraImageInvalid, "AImage_getWidth");
    require_media_ok(AImage_getHeight(image, &height),
                     ErrorCode::CameraImageInvalid, "AImage_getHeight");
    require_media_ok(AImage_getFormat(image, &format),
                     ErrorCode::CameraImageInvalid, "AImage_getFormat");
    require_media_ok(AImage_getTimestamp(image, &timestamp_ns),
                     ErrorCode::CameraImageInvalid, "AImage_getTimestamp");
    if (format != AIMAGE_FORMAT_YUV_420_888 || timestamp_ns <= 0) {
      throw AppError(ErrorCode::CameraImageInvalid,
                     "unexpected image format or timestamp");
    }

    YuvPlaneView planes[3]{};
    for (int plane_index = 0; plane_index < 3; ++plane_index) {
      std::uint8_t* data = nullptr;
      int data_length = 0;
      int32_t row_stride = 0;
      int32_t pixel_stride = 0;
      require_media_ok(
          AImage_getPlaneData(image, plane_index, &data, &data_length),
          ErrorCode::CameraImageInvalid, "AImage_getPlaneData");
      require_media_ok(
          AImage_getPlaneRowStride(image, plane_index, &row_stride),
          ErrorCode::CameraImageInvalid, "AImage_getPlaneRowStride");
      require_media_ok(
          AImage_getPlanePixelStride(image, plane_index, &pixel_stride),
          ErrorCode::CameraImageInvalid, "AImage_getPlanePixelStride");
      planes[plane_index] = YuvPlaneView{
          data, static_cast<std::size_t>(data_length), row_stride, pixel_stride};
    }

    const Yuv420View view{
        width, height, planes[0], planes[1], planes[2]};
    std::vector<std::uint8_t> bgr = yuv420_to_bgr(view);
    if (bgr.empty()) {
      throw AppError(ErrorCode::CameraImageInvalid,
                     "YUV conversion returned no pixels");
    }

    const double timestamp_sec =
        static_cast<double>(timestamp_ns) / 1'000'000'000.0;
    std::uint64_t frame_id = 0;
    {
      std::lock_guard<std::mutex> lock(status_mutex);
      if (snapshot.last_timestamp_sec > 0.0 &&
          timestamp_sec <= snapshot.last_timestamp_sec) {
        ++snapshot.dropped_frames;
        return;
      }
      snapshot.last_timestamp_sec = timestamp_sec;
      frame_id = ++snapshot.accepted_frames;
      if (snapshot.deep_enabled && !snapshot.deep_result_available) {
        snapshot.deep_frames_collected = static_cast<std::size_t>(
            std::min<std::uint64_t>(snapshot.accepted_frames, 180U));
      }
      timestamps.push_back(timestamp_sec);
      while (timestamps.size() > kFpsWindow) {
        timestamps.pop_front();
      }
      if (timestamps.size() >= 2U) {
        const double span = timestamps.back() - timestamps.front();
        if (span > 0.0) {
          snapshot.measured_fps =
              static_cast<double>(timestamps.size() - 1U) / span;
        }
      }
    }

    if (frame_queue != nullptr) {
      cv::Mat wrapped(height, width, CV_8UC3, bgr.data());
      cv::Mat oriented;
      orient_bgr_frame(wrapped, frame_rotation_degrees_, &oriented);
      if (color_diagnostic_requested.exchange(false) &&
          color_diagnostic_queue != nullptr && processing_config.has_value()) {
        ColorDiagnosticJob job;
        job.frame_id = frame_id;
        job.width = width;
        job.height = height;
        job.rotation_degrees = frame_rotation_degrees_;
        job.root = processing_config->output_directory;
        for (std::size_t index = 0; index < job.planes.size(); ++index) {
          job.row_strides[index] = planes[index].row_stride;
          job.pixel_strides[index] = planes[index].pixel_stride;
          job.planes[index].assign(planes[index].data,
                                   planes[index].data + planes[index].size);
        }
        {
          std::lock_guard<std::mutex> lock(status_mutex);
          job.roi_jpeg = latest_roi_jpeg_;
          snapshot.color_diagnostic_state = "writing";
        }
        if (!color_diagnostic_queue->push(std::move(job))) {
          std::lock_guard<std::mutex> lock(status_mutex);
          snapshot.color_diagnostic_state = "failed";
          snapshot.color_diagnostic_finding = "diagnostic_worker_unavailable";
        }
      }
      bool replaced = false;
      if (frame_queue->push(
              FramePacket{frame_id, timestamp_sec, oriented.clone()},
              &replaced) &&
          replaced) {
        std::lock_guard<std::mutex> lock(status_mutex);
        ++snapshot.dropped_frames;
      }
    }
  }

  static void on_device_disconnected(void* context, ACameraDevice*) {
    auto* self = static_cast<Impl*>(context);
    if (self != nullptr) {
      self->set_error(ErrorCode::CameraOpenFailed, "camera disconnected");
    }
  }

  static void on_device_error(void* context, ACameraDevice*, int error) {
    auto* self = static_cast<Impl*>(context);
    if (self != nullptr) {
      self->set_error(ErrorCode::CameraOpenFailed,
                      "camera device error: " + std::to_string(error));
    }
  }

  static void on_session_closed(void*, ACameraCaptureSession*) {}
  static void on_session_ready(void*, ACameraCaptureSession*) {}
  static void on_session_active(void*, ACameraCaptureSession*) {}

  CameraSessionConfig config;
  std::optional<TraditionalProcessingConfig> processing_config;
  std::vector<std::uint8_t> latest_roi_jpeg_;
  double last_roi_jpeg_sec_{0.0};
  int sensor_orientation_{0};
  int display_rotation_degrees_{0};
  int frame_rotation_degrees_{0};
  mutable std::mutex status_mutex;
  CameraSessionStatus snapshot;
  WaveformSnapshot traditional_waveform_;
  WaveformSnapshot deep_waveform_;
  std::uint64_t traditional_waveform_revision_{0};
  std::uint64_t deep_waveform_revision_{0};
  std::deque<double> timestamps;
  std::shared_ptr<LatestFrameQueue> frame_queue;
  std::atomic<bool> processing_stop{false};
  std::thread processing_thread;
  std::atomic<bool> color_diagnostic_requested{false};
  std::unique_ptr<LatestQueue<ColorDiagnosticJob>> color_diagnostic_queue;
  std::thread color_diagnostic_thread;

  std::mutex callback_mutex;
  std::condition_variable callback_condition;
  bool callbacks_enabled{false};
  std::size_t active_callbacks{0};

  ACameraManager* manager{nullptr};
  ACameraDevice* device{nullptr};
  AImageReader* reader{nullptr};
  ::ANativeWindow* window{nullptr};
  ::ANativeWindow* preview_window{nullptr};
  ACaptureSessionOutputContainer* output_container{nullptr};
  ACaptureSessionOutput* session_output{nullptr};
  ACaptureSessionOutput* preview_session_output{nullptr};
  ACameraOutputTarget* output_target{nullptr};
  ACameraOutputTarget* preview_output_target{nullptr};
  ACaptureRequest* request{nullptr};
  ACameraCaptureSession* capture_session{nullptr};
};

ThumbnailRoi::ThumbnailRoi(std::unique_ptr<IRoiProcessor> inner,
                           PublishCallback publish)
    : inner_(std::move(inner)), publish_(std::move(publish)) {}

RoiPacket ThumbnailRoi::process(const FramePacket& frame) {
  RoiPacket packet = inner_->process(frame);
  if (publish_) {
    publish_(packet, frame.bgr.cols, frame.bgr.rows);
  }
  return packet;
}

AndroidCameraSession::AndroidCameraSession(CameraSessionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

AndroidCameraSession::~AndroidCameraSession() = default;

std::vector<std::string> AndroidCameraSession::list_cameras() {
  const std::vector<CameraInfo> infos = list_camera_infos();
  std::vector<std::string> ids;
  ids.reserve(infos.size());
  for (const CameraInfo& info : infos) {
    ids.push_back(info.id);
  }
  return ids;
}

std::vector<CameraInfo> AndroidCameraSession::list_camera_infos() {
  ACameraManager* manager = ACameraManager_create();
  if (manager == nullptr) {
    throw AppError(ErrorCode::CameraOpenFailed,
                   "ACameraManager_create returned null");
  }
  ACameraIdList* camera_ids = nullptr;
  const camera_status_t result =
      ACameraManager_getCameraIdList(manager, &camera_ids);
  if (result != ACAMERA_OK) {
    ACameraManager_delete(manager);
    throw AppError(ErrorCode::CameraOpenFailed,
                   "ACameraManager_getCameraIdList failed: " +
                       std::to_string(result));
  }

  std::vector<CameraInfo> cameras;
  try {
    cameras.reserve(static_cast<std::size_t>(std::max(camera_ids->numCameras, 0)));
    for (int index = 0; index < camera_ids->numCameras; ++index) {
      if (camera_ids->cameraIds[index] == nullptr) {
        continue;
      }
      CameraInfo info;
      info.id = camera_ids->cameraIds[index];
      info.facing = read_lens_facing(manager, info.id.c_str());
      info.sensor_orientation =
          read_sensor_orientation(manager, info.id.c_str());
      cameras.emplace_back(std::move(info));
    }
  } catch (...) {
    ACameraManager_deleteCameraIdList(camera_ids);
    ACameraManager_delete(manager);
    throw;
  }
  ACameraManager_deleteCameraIdList(camera_ids);
  ACameraManager_delete(manager);
  return cameras;
}

void AndroidCameraSession::configure_processing(
    TraditionalProcessingConfig config) {
  impl_->configure_processing(std::move(config));
}

void AndroidCameraSession::set_preview_surface(::ANativeWindow* window) {
  impl_->set_preview_surface(window);
}

void AndroidCameraSession::set_display_rotation(int rotation_degrees) {
  impl_->set_display_rotation(rotation_degrees);
}

void AndroidCameraSession::start() { impl_->start(); }

void AndroidCameraSession::stop() noexcept { impl_->stop(); }

CameraSessionStatus AndroidCameraSession::status() const {
  return impl_->status();
}

std::vector<std::uint8_t> AndroidCameraSession::latest_roi_jpeg() const {
  return impl_->latest_roi_jpeg();
}

void AndroidCameraSession::request_color_diagnostic() {
  impl_->request_color_diagnostic();
}

void AndroidCameraSession::delete_color_diagnostics() {
  impl_->delete_color_diagnostics();
}

WaveformSnapshot AndroidCameraSession::latest_waveform(bool deep) const {
  return impl_->latest_waveform(deep);
}

}  // namespace rppg_qnn::android
