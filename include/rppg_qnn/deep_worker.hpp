#pragma once

#include "rppg_qnn/deep_runtime.hpp"
#include "rppg_qnn/latest_queue.hpp"

#include <memory>
#include <mutex>
#include <optional>
#include <thread>

namespace rppg_qnn {

class DeepWorker {
 public:
  explicit DeepWorker(std::unique_ptr<IDeepRuntime> runtime);
  ~DeepWorker();

  DeepWorker(const DeepWorker&) = delete;
  DeepWorker& operator=(const DeepWorker&) = delete;

  [[nodiscard]] bool submit(DeepInput input);
  [[nodiscard]] std::optional<HeartRateResult> latest_result() const;
  void close();

 private:
  void run();
  void publish(HeartRateResult result);

  std::unique_ptr<IDeepRuntime> runtime_;
  LatestQueue<DeepInput> queue_;
  std::thread thread_;
  mutable std::mutex result_mutex_;
  std::optional<HeartRateResult> latest_;
  std::mutex close_mutex_;
  bool joined_{false};
};

}  // namespace rppg_qnn
