#include "rppg_qnn/deep_worker.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace rppg_qnn {
namespace {

HeartRateResult worker_failure(const DeepInput& input,
                               const std::string& backend,
                               const std::string& detail) {
  HeartRateResult result;
  result.method = "DEEP";
  result.backend = backend;
  result.window_start_sec = std::isfinite(input.start_sec) ? input.start_sec : 0.0;
  result.window_end_sec = std::isfinite(input.end_sec) ? input.end_sec : 0.0;
  result.source_fps = std::isfinite(input.source_fps) ? input.source_fps : 0.0;
  result.source_frame_count = input.source_frame_count;
  result.max_frame_gap_sec =
      std::isfinite(input.max_frame_gap_sec) ? input.max_frame_gap_sec : 0.0;
  result.invalid_reason = "model_inference_failed";
  if (!detail.empty()) {
    result.invalid_reason += ": " + detail;
  }
  return result;
}

}  // namespace

DeepWorker::DeepWorker(std::unique_ptr<IDeepRuntime> runtime,
                       std::chrono::milliseconds idle_wait)
    : runtime_(std::move(runtime)), idle_wait_(idle_wait) {
  if (!runtime_) {
    throw std::invalid_argument("Deep runtime must not be null");
  }
  if (idle_wait_ <= std::chrono::milliseconds::zero()) {
    throw std::invalid_argument("Deep worker idle wait must be positive");
  }
  thread_ = std::thread(&DeepWorker::run, this);
}

DeepWorker::~DeepWorker() { close(); }

bool DeepWorker::submit(DeepInput input) { return queue_.push(std::move(input)); }

std::optional<HeartRateResult> DeepWorker::latest_result() const {
  std::lock_guard<std::mutex> lock(result_mutex_);
  return latest_;
}

void DeepWorker::close() {
  queue_.close();
  std::lock_guard<std::mutex> lock(close_mutex_);
  if (!joined_ && thread_.joinable()) {
    thread_.join();
    joined_ = true;
  }
}

void DeepWorker::run() {
  while (true) {
    auto input = queue_.wait_pop(idle_wait_);
    if (!input.has_value()) {
      if (!queue_.closed()) {
        continue;
      }
      input = queue_.wait_pop(std::chrono::milliseconds::zero());
      if (!input.has_value()) {
        return;
      }
    }
    const auto started = std::chrono::steady_clock::now();
    try {
      publish(runtime_->infer(*input));
    } catch (const std::exception& error) {
      HeartRateResult result =
          worker_failure(*input, runtime_->backend_name(), error.what());
      result.inference_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
      publish(std::move(result));
    } catch (...) {
      HeartRateResult result =
          worker_failure(*input, runtime_->backend_name(), "unknown exception");
      result.inference_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
      publish(std::move(result));
    }
  }
}

void DeepWorker::publish(HeartRateResult result) {
  std::lock_guard<std::mutex> lock(result_mutex_);
  latest_ = std::move(result);
}

}  // namespace rppg_qnn
