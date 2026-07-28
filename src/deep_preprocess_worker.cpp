#include "rppg_qnn/deep_preprocess_worker.hpp"

#include <chrono>
#include <utility>

namespace rppg_qnn {

DeepPreprocessWorker::DeepPreprocessWorker(std::unique_ptr<IDeepRuntime> runtime)
    : inference_(std::move(runtime)), thread_(&DeepPreprocessWorker::run, this) {}

DeepPreprocessWorker::~DeepPreprocessWorker() { close(); }

bool DeepPreprocessWorker::submit(DeepWindowBuilder builder) {
  const bool replaced = queue_.size() > 0U;
  const bool accepted = queue_.push(std::move(builder));
  if (accepted) {
    std::lock_guard<std::mutex> lock(metrics_mutex_);
    ++metrics_.submitted;
    if (replaced) ++metrics_.replaced;
  }
  return accepted;
}

std::optional<HeartRateResult> DeepPreprocessWorker::latest_result() const {
  return inference_.latest_result();
}

DeepPreprocessMetrics DeepPreprocessWorker::metrics() const {
  std::lock_guard<std::mutex> lock(metrics_mutex_);
  return metrics_;
}

void DeepPreprocessWorker::close() {
  queue_.close();
  std::lock_guard<std::mutex> lock(close_mutex_);
  if (!joined_ && thread_.joinable()) {
    thread_.join();
    joined_ = true;
  }
  inference_.close();
}

void DeepPreprocessWorker::run() {
  while (true) {
    auto builder = queue_.wait_pop(std::chrono::hours(24));
    if (!builder.has_value()) {
      if (!queue_.closed()) continue;
      builder = queue_.wait_pop(std::chrono::milliseconds::zero());
      if (!builder.has_value()) return;
    }
    const auto started = std::chrono::steady_clock::now();
    std::optional<DeepInput> input = builder->build_latest();
    const double elapsed = std::chrono::duration<double, std::milli>(
                               std::chrono::steady_clock::now() - started)
                               .count();
    {
      std::lock_guard<std::mutex> lock(metrics_mutex_);
      metrics_.latest_ms = elapsed;
      metrics_.status = builder->status();
      if (input.has_value()) ++metrics_.materialized;
    }
    if (input.has_value()) {
      (void)inference_.submit(std::move(*input));
    }
  }
}

}  // namespace rppg_qnn
