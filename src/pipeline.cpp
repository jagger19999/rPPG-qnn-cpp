#include "rppg_qnn/pipeline.hpp"

#include "rppg_qnn/deep_window_builder.hpp"
#include "rppg_qnn/deep_worker.hpp"
#include "rppg_qnn/error.hpp"
#include "rppg_qnn/green_predictor.hpp"
#include "rppg_qnn/qnn_preflight.hpp"
#include "rppg_qnn/result_sink.hpp"

#include <cmath>
#include <deque>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace rppg_qnn {
namespace {

int exit_code_for(const AppError& error) {
  switch (error.code()) {
    case ErrorCode::ConfigInvalid: return 2;
    case ErrorCode::CameraOpenFailed: return 3;
    case ErrorCode::CameraFormatUnsupported: return 4;
    case ErrorCode::LowCaptureFps: return 5;
    case ErrorCode::FaceNotFound: return 6;
    case ErrorCode::QnnLibraryNotFound: return 7;
    case ErrorCode::QnnApiIncompatible: return 8;
    case ErrorCode::QnnGpuInitFailed: return 9;
    case ErrorCode::ModelManifestInvalid: return 10;
    case ErrorCode::ModelLoadFailed: return 11;
    case ErrorCode::InferenceFailed: return 12;
    case ErrorCode::OutputWriteFailed: return 13;
  }
  return 1;
}

int error_exit_code(const std::exception& error) {
  const auto* app_error = dynamic_cast<const AppError*>(&error);
  return app_error == nullptr ? 1 : exit_code_for(*app_error);
}

std::string error_code(const std::exception& error) {
  const auto* app_error = dynamic_cast<const AppError*>(&error);
  return app_error == nullptr ? "UNEXPECTED_EXCEPTION"
                              : std::string(to_string(app_error->code()));
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

}  // namespace

Pipeline::Pipeline(AppConfig config, PipelineDependencies dependencies)
    : config_(std::move(config)), dependencies_(std::move(dependencies)) {}

int Pipeline::run() {
  std::unique_ptr<ResultSink> sink;
  std::unique_ptr<DeepWorker> worker;
  try {
    sink = std::make_unique<ResultSink>(config_.output);
    sink->publish(run_qnn_preflight(config_));
    if (config_.preflight_only) {
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
    std::optional<double> last_submitted_deep_end;
    while (const std::optional<FramePacket> frame = source->read()) {
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
            sink->publish(*result);
          }
        }
        if (deep_builder.has_value()) {
          const std::optional<DeepInput> input = deep_builder->add_roi(roi);
          health.status = deep_builder->status();
          if (input.has_value() &&
              (!last_submitted_deep_end.has_value() ||
               input->end_sec - *last_submitted_deep_end >= 1.0)) {
            last_submitted_deep_end = input->end_sec;
            if (!worker->submit(*input)) {
              health.status = "deep_submit_closed";
            }
          }
        }
      }
      sink->publish(health);
      const std::optional<HeartRateResult> deep_result =
          worker ? worker->latest_result() : std::optional<HeartRateResult>{};
      if (is_new_window(deep_result, &published_deep_end)) {
        sink->publish(*deep_result);
      }
    }

    if (worker) {
      worker->close();
      const std::optional<HeartRateResult> result = worker->latest_result();
      if (is_new_window(result, &published_deep_end)) {
        sink->publish(*result);
      }
    }
    sink->close(0);
    return 0;
  } catch (const std::exception& error) {
    if (worker) {
      worker->close();
    }
    const int exit_code = error_exit_code(error);
    if (sink) {
      try {
        sink->publish_runtime_error(error_code(error), error.what());
      } catch (...) {
      }
      try {
        sink->close(exit_code);
      } catch (...) {
      }
    }
    return exit_code;
  }
}

}  // namespace rppg_qnn
