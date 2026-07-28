#pragma once

#include "rppg_qnn/deep_runtime.hpp"
#include "rppg_qnn/deep_window_builder.hpp"
#include "rppg_qnn/deep_worker.hpp"
#include "rppg_qnn/latest_queue.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>

namespace rppg_qnn {

struct DeepPreprocessMetrics {
  double latest_ms{0.0};
  std::size_t submitted{0};
  std::size_t replaced{0};
  std::size_t materialized{0};
  std::string status{"sampling"};
};

class DeepPreprocessWorker {
 public:
  explicit DeepPreprocessWorker(std::unique_ptr<IDeepRuntime> runtime);
  ~DeepPreprocessWorker();

  DeepPreprocessWorker(const DeepPreprocessWorker&) = delete;
  DeepPreprocessWorker& operator=(const DeepPreprocessWorker&) = delete;

  bool submit(DeepWindowBuilder builder);
  [[nodiscard]] std::optional<HeartRateResult> latest_result() const;
  [[nodiscard]] DeepPreprocessMetrics metrics() const;
  void close();

 private:
  void run();

  DeepWorker inference_;
  LatestQueue<DeepWindowBuilder> queue_;
  std::thread thread_;
  mutable std::mutex metrics_mutex_;
  DeepPreprocessMetrics metrics_;
  std::mutex close_mutex_;
  bool joined_{false};
};

}  // namespace rppg_qnn
