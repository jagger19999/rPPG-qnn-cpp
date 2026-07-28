#include "rppg_qnn/pipeline.hpp"

#include "rppg_qnn/deep_window_builder.hpp"
#include "rppg_qnn/deep_preprocess_worker.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/qnn_preflight.hpp"
#include "rppg_qnn/traditional_ensemble.hpp"

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
#include <vector>

namespace rppg_qnn {
namespace {

constexpr std::size_t kMaxEmptyReads = 3U;
constexpr std::size_t kMaxOutputEvents = 256U;

class RollingDurations {
 public:
  TimingDistribution add(double milliseconds) {
    values_.push_back(milliseconds);
    while (values_.size() > 120U) values_.pop_front();
    std::vector<double> sorted(values_.begin(), values_.end());
    std::sort(sorted.begin(), sorted.end());
    const auto percentile = [&sorted](double fraction) {
      if (sorted.empty()) return 0.0;
      const std::size_t index = static_cast<std::size_t>(
          std::floor(fraction * static_cast<double>(sorted.size() - 1U)));
      return sorted[index];
    };
    return {milliseconds, percentile(0.50), percentile(0.95)};
  }
 private:
  std::deque<double> values_;
};

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
  explicit AsyncOutputWorker(IResultSink& sink) : sink_(sink) {
    thread_ = std::thread(&AsyncOutputWorker::run, this);
  }
  ~AsyncOutputWorker() { close_noexcept(); }

  AsyncOutputWorker(const AsyncOutputWorker&) = delete;
  AsyncOutputWorker& operator=(const AsyncOutputWorker&) = delete;

  bool submit(FrameHealth health) { return submit_health(std::move(health)); }
  bool submit(HeartRateResult result) { return submit_result(std::move(result)); }
  bool submit(MeasurementSnapshot snapshot) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || closing_ || failure_ != nullptr ||
        events_.size() >= kMaxOutputEvents - 1U) {
      return false;
    }
    OutputEvent event;
    event.kind = OutputEvent::Kind::Measurement;
    event.measurement = std::move(snapshot);
    events_.push_back(std::move(event));
    condition_.notify_one();
    return true;
  }
  bool submit_runtime_error(std::string code, std::string message) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || closing_ || failure_ != nullptr || events_.size() >= kMaxOutputEvents) {
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
    std::lock_guard<std::mutex> close_lock(close_mutex_);
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (!closed_ && !closing_) {
        closing_ = true;
        condition_.notify_one();
      }
    }
    if (thread_.joinable()) {
      thread_.join();
    }
    std::exception_ptr failure;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      closed_ = true;
      failure = failure_;
    }
    if (failure != nullptr) {
      try {
        std::rethrow_exception(failure);
      } catch (const std::exception& error) {
        throw AppError(ErrorCode::OutputWriteFailed, error.what());
      } catch (...) {
        throw AppError(ErrorCode::OutputWriteFailed, "non-standard output failure");
      }
    }
  }

 private:
  struct OutputEvent {
    enum class Kind { HeartRate, Measurement, RuntimeError };
    Kind kind{Kind::HeartRate};
    HeartRateResult heart_rate;
    MeasurementSnapshot measurement;
    std::string error_code;
    std::string error_message;
  };

  bool submit_health(FrameHealth health) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || closing_ || failure_ != nullptr) {
      return false;
    }
    latest_health_ = std::move(health);
    condition_.notify_one();
    return true;
  }

  bool submit_result(HeartRateResult result) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (closed_ || closing_ || failure_ != nullptr ||
        events_.size() >= kMaxOutputEvents - 1U) {
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
          return closing_ || failure_ != nullptr || latest_health_.has_value() || !events_.empty();
        });
        if (failure_ != nullptr) {
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
          } else if (event->kind == OutputEvent::Kind::Measurement) {
            sink_.publish(event->measurement);
          } else {
            sink_.publish_runtime_error(event->error_code, event->error_message);
          }
        } else if (health.has_value()) {
          sink_.publish(*health);
        }
      } catch (const std::exception&) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_ = std::current_exception();
        condition_.notify_all();
        return;
      } catch (...) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_ = std::current_exception();
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
  std::mutex close_mutex_;
  std::optional<FrameHealth> latest_health_;
  std::deque<OutputEvent> events_;
  bool closing_{false};
  bool closed_{false};
  std::exception_ptr failure_;
  std::thread thread_;
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
  std::unique_ptr<DeepPreprocessWorker> worker;
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
    if (config_.deep != "disabled") {
      if (!dependencies_.make_deep_runtime) {
        throw AppError(ErrorCode::ConfigInvalid,
                       "Requested deep runtime factory is required");
      }
      std::unique_ptr<IDeepRuntime> runtime = dependencies_.make_deep_runtime();
      if (!runtime) {
        throw AppError(ErrorCode::ConfigInvalid, "Deep runtime factory returned null");
      }
      deep_builder.emplace(6.0, 180U, cv::Size(72, 72));
      worker = std::make_unique<DeepPreprocessWorker>(std::move(runtime));
    }

    TraditionalEnsemble traditional;
    std::deque<double> timestamps;
    std::size_t published_traditional_evaluations = 0;
    std::optional<cv::Point2d> previous_face_center;
    std::optional<double> published_deep_end;
    std::optional<double> last_deep_build_sec;
    RollingDurations roi_timings;
    RollingDurations traditional_timings;
    RollingDurations deep_preprocess_timings;
    std::deque<std::chrono::steady_clock::time_point> processing_timestamps;
    std::size_t empty_reads = 0;
    while (true) {
      if (dependencies_.should_stop && dependencies_.should_stop()) {
        break;
      }
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
      const auto roi_started = std::chrono::steady_clock::now();
      RoiPacket roi = roi_processor->process(*frame);
      const double roi_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - roi_started)
                                .count();
      FrameHealth health;
      health.frame_id = frame->frame_id;
      health.timestamp_sec = frame->timestamp_sec;
      health.capture_fps = capture_fps(&timestamps, frame->timestamp_sec);
      health.face_found = roi.face.has_value() && !roi.roi_bgr.empty();
      health.face_confidence = roi.face.has_value() ? roi.face->confidence : 0.0;
      health.status = health.face_found ? "ok" : "face_not_found";
      health.performance.roi = roi_timings.add(roi_ms);
      const auto processing_now = std::chrono::steady_clock::now();
      processing_timestamps.push_back(processing_now);
      while (!processing_timestamps.empty() &&
             processing_now - processing_timestamps.front() >
                 std::chrono::seconds(4)) {
        processing_timestamps.pop_front();
      }
      if (processing_timestamps.size() >= 2U) {
        const double seconds = std::chrono::duration<double>(
                                   processing_timestamps.back() -
                                   processing_timestamps.front())
                                   .count();
        health.performance.processing_fps =
            seconds > 0.0
                ? static_cast<double>(processing_timestamps.size() - 1U) / seconds
                : 0.0;
      }

      if (health.face_found) {
        const auto traditional_started = std::chrono::steady_clock::now();
        FrameQualitySample quality_sample;
        quality_sample.face_count = roi.face_count;
        quality_sample.motion_px = roi.motion_px;
        if (roi.face.has_value() && frame->bgr.cols > 0 && frame->bgr.rows > 0) {
          quality_sample.face_area_ratio =
              static_cast<double>(roi.face->width * roi.face->height) /
              static_cast<double>(frame->bgr.cols * frame->bgr.rows);
          const cv::Point2d center(
              static_cast<double>(roi.face->x) + roi.face->width / 2.0,
              static_cast<double>(roi.face->y) + roi.face->height / 2.0);
          if (quality_sample.motion_px <= 0.0 && previous_face_center.has_value()) {
            quality_sample.motion_px = cv::norm(center - *previous_face_center);
          }
          previous_face_center = center;
        }
        traditional.add_sample(roi.timestamp_sec, cv::mean(roi.roi_bgr),
                               quality_sample);
        health.performance.traditional = traditional_timings.add(
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - traditional_started)
                .count());
        if (traditional.evaluation_count() != published_traditional_evaluations) {
          published_traditional_evaluations = traditional.evaluation_count();
          const std::optional<MeasurementSnapshot> snapshot =
              traditional.latest_snapshot();
          if (snapshot.has_value()) {
            require_output(output->submit(*snapshot));
            const std::string configured_method = traditional_method_name(
                traditional_method_from_string(config_.traditional));
            const auto selected = std::find_if(
                snapshot->candidates.begin(), snapshot->candidates.end(),
                [&configured_method](const CandidateResult& candidate) {
                  return candidate.method == configured_method;
                });
            if (selected != snapshot->candidates.end()) {
              HeartRateResult result;
              result.method = selected->method;
              result.window_start_sec = snapshot->window_end_sec - 10.0;
              result.window_end_sec = snapshot->window_end_sec;
              result.bpm = selected->bpm;
              result.confidence = selected->confidence;
              result.peak_ratio = selected->peak_ratio;
              result.is_valid = selected->valid;
              result.invalid_reason = selected->invalid_reason;
              result.source_fps = snapshot->quality.source_fps;
              result.max_frame_gap_sec = snapshot->quality.max_frame_gap_sec;
              result.waveform = selected->waveform;
              result.backend = "cpu";
              require_output(output->submit(std::move(result)));
            }
          }
        }
        if (worker) {
          const DeepPreprocessMetrics metrics = worker->metrics();
          health.performance.deep_preprocess =
              deep_preprocess_timings.add(metrics.latest_ms);
        }
        if (deep_builder.has_value()) {
          RoiPacket deep_packet = roi;
          if (!roi.deep_roi_bgr.empty()) {
            deep_packet.roi_bgr = roi.deep_roi_bgr;
          }
          const bool ingested = deep_builder->ingest_roi(deep_packet);
          health.status = deep_builder->status();
          if (ingested && (!last_deep_build_sec.has_value() ||
                           roi.timestamp_sec - *last_deep_build_sec >= 1.0)) {
            last_deep_build_sec = roi.timestamp_sec;
            require_output(worker->submit(*deep_builder));
            health.status = "deep_preprocess_pending";
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
