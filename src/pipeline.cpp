#include "rppg_qnn/pipeline.hpp"

#include "rppg_qnn/deep_window_builder.hpp"
#include "rppg_qnn/deep_worker.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/green_predictor.hpp"
#include "rppg_qnn/qnn_preflight.hpp"

#include <chrono>
#include <cmath>
#include <condition_variable>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

namespace rppg_qnn {
namespace {

constexpr std::size_t kMaxEmptyReads = 3U;
constexpr std::size_t kMaxOutputEvents = 256U;

int error_exit_code(const std::exception& error) {
  const auto* app_error = dynamic_cast<const AppError*>(&error);
  return app_error == nullptr ? 1 : exit_code_for(app_error->code());
}

std::string error_code(const std::exception& error) {
  const auto* app_error = dynamic_cast<const AppError*>(&error);
  return app_error == nullptr ? "UNEXPECTED_EXCEPTION"
                              : std::string(to_string(app_error->code()));
}

AppError preflight_error(const PreflightResult& result) {
  if (result.error.find("QNN_LIBRARY_NOT_FOUND") != std::string::npos) {
    return AppError(ErrorCode::QnnLibraryNotFound, result.error);
  }
  if (result.error.find("QNN_API_INCOMPATIBLE") != std::string::npos) {
    return AppError(ErrorCode::QnnApiIncompatible, result.error);
  }
  return AppError(ErrorCode::QnnGpuInitFailed, result.error);
}

double capture_fps(std::deque<double>* timestamps, double timestamp_sec) {
  if (!std::isfinite(timestamp_sec)) {
    return 0.0;
  }
  timestamps->push_back(timestamp_sec);
  while (timestamps->size() > 120U) {
    timestamps->pop_front();
  }
  if (timestamps->size() < 2U) {
    return 0.0;
  }
  const double span = timestamps->back() - timestamps->front();
  return span > 0.0 ? static_cast<double>(timestamps->size() - 1U) / span : 0.0;
}

bool is_new_window(const std::optional<HeartRateResult>& result,
                   std::optional<double>* published_end) {
  if (!result.has_value() || !std::isfinite(result->window_end_sec) ||
      (published_end->has_value() && result->window_end_sec <= **published_end)) {
    return false;
  }
  *published_end = result->window_end_sec;
  return true;
}

class AsyncOutputWorker {
 public:
  explicit AsyncOutputWorker(IResultSink& sink) : sink_(sink), thread_(&AsyncOutputWorker::run, this) {}
  ~AsyncOutputWorker() { close_noexcept(); }

  AsyncOutputWorker(const AsyncOutputWorker&) = delete;
  AsyncOutputWorker& operator=(const AsyncOutputWorker&) = delete;

  bool submit(FrameHealth health) { return submit_health(std::move(health)); }
  bool submit(HeartRateResult result) { return submit_result(std::move(result)); }
  bool submit_runtime_error(std::string code, std::string message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || !failure_.empty() || events_.size() >= kMaxOutputEvents) {
      return false;
    }
    OutputEvent event;
    event.kind = OutputEvent::Kind::RuntimeError;
    event.error_code = std::move(code);
    event.error_message = std::move(message);
    events_.push_back(std::move(event));
    condition_.notify_one();
    return true;
  }

  void close() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!closed_) {
        closing_ = true;
        condition_.notify_one();
      }
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    closed_ = true;
    std::lock_guard<std::mutex> lock(mutex_);
    if (!failure_.empty()) {
      throw AppError(ErrorCode::OutputWriteFailed, failure_);
    }
  }

 private:
  struct OutputEvent {
    enum class Kind { HeartRate, RuntimeError };
    Kind kind{Kind::HeartRate};
    HeartRateResult heart_rate;
    std::string error_code;
    std::string error_message;
  };

  bool submit_health(FrameHealth health) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || !failure_.empty()) {
      return false;
    }
    latest_health_ = std::move(health);
    condition_.notify_one();
    return true;
  }

  bool submit_result(HeartRateResult result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || !failure_.empty() || events_.size() >= kMaxOutputEvents) {
      return false;
    }
    OutputEvent event;
    event.kind = OutputEvent::Kind::HeartRate;
    event.heart_rate = std::move(result);
    events_.push_back(std::move(event));
    condition_.notify_one();
    return true;
  }

  void run() {
    while (true) {
      std::optional<FrameHealth> health;
      std::optional<OutputEvent> event;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] {
          return closing_ || !failure_.empty() || latest_health_.has_value() || !events_.empty();
        });
        if (!failure_.empty()) {
          return;
        }
        if (!events_.empty()) {
          event = std::move(events_.front());
          events_.pop_front();
        } else if (latest_health_.has_value()) {
          health = std::move(latest_health_);
          latest_health_.reset();
        } else if (closing_) {
          return;
        }
      }
      try {
        if (event.has_value()) {
          if (event->kind == OutputEvent::Kind::HeartRate) {
            sink_.publish(event->heart_rate);
          } else {
            sink_.publish_runtime_error(event->error_code, event->error_message);
          }
        } else if (health.has_value()) {
          sink_.publish(*health);
        }
      } catch (const std::exception& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_ = error.what();
        condition_.notify_all();
        return;
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_ = "non-standard output failure";
        condition_.notify_all();
        return;
      }
    }
  }

  void close_noexcept() noexcept {
    try {
      close();
    } catch (...) {
    }
  }

  IResultSink& sink_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<FrameHealth> latest_health_;
  std::deque<OutputEvent> events_;
  std::thread thread_;
  bool closing_{false};
  bool closed_{false};
  std::string failure_;
};

void require_output(bool submitted) {
  if (!submitted) {
    throw AppError(ErrorCode::OutputWriteFailed, "asynchronous output worker unavailable");
  }
}

}  // namespace

Pipeline::Pipeline(AppConfig config, PipelineDependencies dependencies)
    : config_(std::move(config)), dependencies_(std::move(dependencies)) {}

int Pipeline::run() {
  std::unique_ptr<IResultSink> sink;
  std::unique_ptr<AsyncOutputWorker> output;
  std::unique_ptr<DeepWorker> worker;
  bool preflight_failure_reported{false};
  try {
    sink = dependencies_.make_sink ? dependencies_.make_sink(config_.output)
                                   : std::make_unique<ResultSink>(config_.output);
    if (!sink) {
      throw AppError(ErrorCode::OutputWriteFailed, "result sink factory returned null");
    }
    const PreflightResult preflight = run_qnn_preflight(config_);
    sink->publish(preflight);
    if (config_.preflight_only) {
      if (!preflight.qnn_gpu_available) {
        preflight_failure_reported = true;
        throw preflight_error(preflight);
      }
      sink->close(0);
      return 0;
    }
    if (!dependencies_.make_source || !dependencies_.make_roi) {
      throw AppError(ErrorCode::ConfigInvalid, "Pipeline source and ROI factories are required");
    }
    std::unique_ptr<FrameSource> source = dependencies_.make_source();
    std::unique_ptr<IRoiProcessor> roi_processor = dependencies_.make_roi();
    if (!source || !roi_processor) {
      throw AppError(ErrorCode::ConfigInvalid, "Pipeline factories must not return null");
    }
    output = std::make_unique<AsyncOutputWorker>(*sink);

    std::optional<DeepWindowBuilder> deep_builder;
    if (config_.deep == "fake") {
      if (!dependencies_.make_deep_runtime) {
        throw AppError(ErrorCode::ConfigInvalid, "Fake deep runtime factory is required");
      }
      std::unique_ptr<IDeepRuntime> runtime = dependencies_.make_deep_runtime();
      if (!runtime) {
        throw AppError(ErrorCode::ConfigInvalid, "Deep runtime factory returned null");
      }
      deep_builder.emplace(6.0, 180U, cv::Size(72, 72));
      worker = std::make_unique<DeepWorker>(std::move(runtime));
    } else if (config_.deep != "disabled") {
      throw AppError(ErrorCode::ConfigInvalid, "Unsupported deep runtime: " + config_.deep);
    }

    GreenPredictor green;
    std::deque<double> timestamps;
    std::size_t published_green_evaluations = 0;
    std::optional<double> published_deep_end;
    std::optional<double> last_deep_build_sec;
    std::size_t empty_reads = 0;
    while (true) {
      std::optional<FramePacket> frame = source->read();
      if (!frame.has_value()) {
        if (source->eof()) {
          break;
        }
        if (++empty_reads > kMaxEmptyReads) {
          throw AppError(ErrorCode::CameraOpenFailed, "too many consecutive empty frame reads");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        continue;
      }
      empty_reads = 0;
      RoiPacket roi = roi_processor->process(*frame);
      FrameHealth health;
      health.frame_id = frame->frame_id;
      health.timestamp_sec = frame->timestamp_sec;
      health.capture_fps = capture_fps(&timestamps, frame->timestamp_sec);
      health.face_found = roi.face.has_value() && !roi.roi_bgr.empty();
      health.face_confidence = roi.face.has_value() ? roi.face->confidence : 0.0;
      health.status = health.face_found ? "ok" : "face_not_found";

      if (health.face_found) {
        green.add_sample(roi.timestamp_sec, cv::mean(roi.roi_bgr));
        if (green.evaluation_count() != published_green_evaluations) {
          published_green_evaluations = green.evaluation_count();
          const std::optional<HeartRateResult> result = green.latest_result();
          if (result.has_value()) {
            require_output(output->submit(*result));
          }
        }
        if (deep_builder.has_value()) {
          (void)deep_builder->ingest_roi(roi);
          health.status = deep_builder->status();
          if (!last_deep_build_sec.has_value() ||
              roi.timestamp_sec - *last_deep_build_sec >= 1.0) {
            last_deep_build_sec = roi.timestamp_sec;
            std::optional<DeepInput> input = deep_builder->build_latest();
            health.status = deep_builder->status();
            if (input.has_value()) {
              require_output(worker->submit(std::move(*input)));
            }
          }
        }
      }
      require_output(output->submit(std::move(health)));
      const std::optional<HeartRateResult> deep_result =
          worker ? worker->latest_result() : std::optional<HeartRateResult>{};
      if (is_new_window(deep_result, &published_deep_end)) {
        require_output(output->submit(*deep_result));
      }
    }

    if (worker) {
      worker->close();
      const std::optional<HeartRateResult> result = worker->latest_result();
      if (is_new_window(result, &published_deep_end)) {
        require_output(output->submit(*result));
      }
    }
    output->close();
    sink->close(0);
    return 0;
  } catch (const std::exception& error) {
    std::string cleanup_failure;
    if (worker) {
      try {
        worker->close();
      } catch (const std::exception& cleanup_error) {
        cleanup_failure = cleanup_error.what();
      }
    }
    const int exit_code = error_exit_code(error);
    if (output) {
      if (!preflight_failure_reported) {
        (void)output->submit_runtime_error(error_code(error), error.what());
      }
      try {
        output->close();
      } catch (const std::exception& cleanup_error) {
        cleanup_failure = cleanup_error.what();
      }
    } else if (sink && !preflight_failure_reported) {
      try {
        sink->publish_runtime_error(error_code(error), error.what());
      } catch (const std::exception& cleanup_error) {
        cleanup_failure = cleanup_error.what();
      }
    }
    if (sink) {
      try {
        sink->close(exit_code);
      } catch (const std::exception& cleanup_error) {
        cleanup_failure = cleanup_error.what();
      }
    }
    std::string message = error.what();
    if (!cleanup_failure.empty()) {
      message += "; output cleanup failed: " + cleanup_failure;
    }
    print_error_line(error_code(error), message);
    return exit_code;
  }
}

}  // namespace rppg_qnn
