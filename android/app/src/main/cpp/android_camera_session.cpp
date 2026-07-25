#include "android_camera_session.hpp"
#include "android_onnx_cpu_runtime.hpp"

#include "rppg_qnn/config.hpp"
#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/frame_source.hpp"
#include "rppg_qnn/pipeline.hpp"
#include "rppg_qnn/result_sink.hpp"
#include "rppg_qnn/roi_processor.hpp"
#include "rppg_qnn/yuv420.hpp"

#include <android/log.h>
#include <camera/NdkCameraDevice.h>
#include <camera/NdkCameraManager.h>
#include <media/NdkImage.h>
#include <media/NdkImageReader.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
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
  using ErrorCallback =
      std::function<void(const std::string&, const std::string&)>;

  StatusResultSink(const std::filesystem::path& output_directory,
                   FrameCallback frame_callback,
                   HeartRateCallback heart_rate_callback,
                   ErrorCallback error_callback)
      : sink_(output_directory),
        frame_callback_(std::move(frame_callback)),
        heart_rate_callback_(std::move(heart_rate_callback)),
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

  void close(int exit_code) override { sink_.close(exit_code); }

 private:
  ResultSink sink_;
  FrameCallback frame_callback_;
  HeartRateCallback heart_rate_callback_;
  ErrorCallback error_callback_;
};

class ThumbnailRoi final : public IRoiProcessor {
 public:
  using PublishCallback = std::function<void(const RoiPacket&)>;

  ThumbnailRoi(std::unique_ptr<IRoiProcessor> inner, PublishCallback publish);

  RoiPacket process(const FramePacket& frame) override;

 private:
  std::unique_ptr<IRoiProcessor> inner_;
  PublishCallback publish_;
};

}  // namespace

struct AndroidCameraSession::Impl {
  explicit Impl(CameraSessionConfig requested_config)
      : config(std::move(requested_config)) {
    snapshot.camera_id = config.camera_id;
    snapshot.requested_width = config.width;
    snapshot.requested_height = config.height;
    snapshot.requested_fps = config.fps;
  }

  ~Impl() { stop(); }

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
    if (requested_config.deep_enabled &&
        (requested_config.model_path.empty() ||
         !std::filesystem::is_regular_file(requested_config.model_path))) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     "EfficientPhys ONNX model is missing from app storage");
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
    snapshot.deep_enabled = processing_config->deep_enabled;
    snapshot.deep_backend =
        processing_config->deep_enabled ? "ONNX_RUNTIME_CPU" : "disabled";
  }

  void start_processing() {
    if (!processing_config.has_value()) {
      return;
    }

    const TraditionalProcessingConfig requested = *processing_config;
    frame_queue = std::make_shared<LatestFrameQueue>();
    processing_stop.store(false);

    AppConfig pipeline_config;
    pipeline_config.camera = config.camera_id;
    pipeline_config.width = config.width;
    pipeline_config.height = config.height;
    pipeline_config.fps = static_cast<double>(config.fps);
    pipeline_config.traditional = requested.method;
    pipeline_config.deep =
        requested.deep_enabled ? "onnxruntime_cpu" : "disabled";
    pipeline_config.output = requested.output_directory;

    PipelineDependencies dependencies;
    const std::shared_ptr<LatestFrameQueue> queue = frame_queue;
    dependencies.make_source = [queue, fps = pipeline_config.fps] {
      return std::make_unique<QueueFrameSource>(queue, fps);
    };
    dependencies.make_roi = [this, cascade = requested.cascade_path] {
      ThumbnailRoi::PublishCallback publish =
          [this](const RoiPacket& packet) { maybe_publish_roi_jpeg(packet); };
      return std::make_unique<ThumbnailRoi>(
          std::make_unique<RoiProcessor>(cascade), std::move(publish));
    };
    if (requested.deep_enabled) {
      dependencies.make_deep_runtime = [model = requested.model_path] {
        return make_onnx_cpu_runtime(model);
      };
    }
    dependencies.make_sink = [this](
                                 const std::filesystem::path& output_directory) {
      return std::make_unique<StatusResultSink>(
          output_directory,
          [this](const FrameHealth& health) {
            std::lock_guard<std::mutex> lock(status_mutex);
            snapshot.face_found = health.face_found;
          },
          [this](const HeartRateResult& result) {
            std::lock_guard<std::mutex> lock(status_mutex);
            if (result.method == "EFFICIENTPHYS" || result.method == "DEEP") {
              snapshot.deep_result_available = true;
              snapshot.deep_bpm = result.bpm;
              snapshot.deep_confidence = result.confidence;
              snapshot.deep_inference_ms = result.inference_ms;
              snapshot.deep_result_valid = result.is_valid;
              snapshot.deep_invalid_reason = result.invalid_reason;
            } else {
              snapshot.heart_rate_available = true;
              snapshot.bpm = result.bpm;
              snapshot.confidence = result.confidence;
              snapshot.heart_rate_valid = result.is_valid;
              snapshot.heart_rate_invalid_reason = result.invalid_reason;
              snapshot.window_start_sec = result.window_start_sec;
              snapshot.window_end_sec = result.window_end_sec;
            }
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
    frame_queue.reset();
    clear_roi_jpeg();
  }

  void clear_roi_jpeg() {
    std::lock_guard<std::mutex> lock(status_mutex);
    latest_roi_jpeg_.clear();
    last_roi_jpeg_sec_ = 0.0;
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
      snapshot.deep_inference_ms = 0.0;
      snapshot.deep_result_valid = false;
      snapshot.deep_invalid_reason.clear();
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

      ACameraCaptureSession_stateCallbacks session_callbacks{
          this, &Impl::on_session_closed, &Impl::on_session_ready,
          &Impl::on_session_active};
      require_camera_ok(ACameraDevice_createCaptureSession(
                            device, output_container, &session_callbacks,
                            &capture_session),
                        ErrorCode::CameraOpenFailed,
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
    if (output_container != nullptr && session_output != nullptr) {
      (void)ACaptureSessionOutputContainer_remove(output_container,
                                                  session_output);
    }
    if (session_output != nullptr) {
      ACaptureSessionOutput_free(session_output);
      session_output = nullptr;
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
      bool replaced = false;
      if (frame_queue->push(
              FramePacket{frame_id, timestamp_sec, wrapped.clone()},
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
  mutable std::mutex status_mutex;
  CameraSessionStatus snapshot;
  std::deque<double> timestamps;
  std::shared_ptr<LatestFrameQueue> frame_queue;
  std::atomic<bool> processing_stop{false};
  std::thread processing_thread;

  std::mutex callback_mutex;
  std::condition_variable callback_condition;
  bool callbacks_enabled{false};
  std::size_t active_callbacks{0};

  ACameraManager* manager{nullptr};
  ACameraDevice* device{nullptr};
  AImageReader* reader{nullptr};
  ANativeWindow* window{nullptr};
  ACaptureSessionOutputContainer* output_container{nullptr};
  ACaptureSessionOutput* session_output{nullptr};
  ACameraOutputTarget* output_target{nullptr};
  ACaptureRequest* request{nullptr};
  ACameraCaptureSession* capture_session{nullptr};
};

ThumbnailRoi::ThumbnailRoi(std::unique_ptr<IRoiProcessor> inner,
                           PublishCallback publish)
    : inner_(std::move(inner)), publish_(std::move(publish)) {}

RoiPacket ThumbnailRoi::process(const FramePacket& frame) {
  RoiPacket packet = inner_->process(frame);
  if (publish_) {
    publish_(packet);
  }
  return packet;
}

AndroidCameraSession::AndroidCameraSession(CameraSessionConfig config)
    : impl_(std::make_unique<Impl>(std::move(config))) {}

AndroidCameraSession::~AndroidCameraSession() = default;

std::vector<std::string> AndroidCameraSession::list_cameras() {
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

  std::vector<std::string> ids;
  try {
    ids.reserve(static_cast<std::size_t>(std::max(camera_ids->numCameras, 0)));
    for (int index = 0; index < camera_ids->numCameras; ++index) {
      if (camera_ids->cameraIds[index] != nullptr) {
        ids.emplace_back(camera_ids->cameraIds[index]);
      }
    }
  } catch (...) {
    ACameraManager_deleteCameraIdList(camera_ids);
    ACameraManager_delete(manager);
    throw;
  }
  ACameraManager_deleteCameraIdList(camera_ids);
  ACameraManager_delete(manager);
  return ids;
}

void AndroidCameraSession::configure_processing(
    TraditionalProcessingConfig config) {
  impl_->configure_processing(std::move(config));
}

void AndroidCameraSession::start() { impl_->start(); }

void AndroidCameraSession::stop() noexcept { impl_->stop(); }

CameraSessionStatus AndroidCameraSession::status() const {
  return impl_->status();
}

std::vector<std::uint8_t> AndroidCameraSession::latest_roi_jpeg() const {
  return impl_->latest_roi_jpeg();
}

}  // namespace rppg_qnn::android
