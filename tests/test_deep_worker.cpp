#include "rppg_qnn/deep_worker.hpp"

#include "test_support.hpp"

#include <chrono>
#include <cmath>
#include <stdexcept>
#include <thread>
#include <utility>

namespace {

using namespace std::chrono_literals;

rppg_qnn::DeepInput valid_input(double end_sec) {
  rppg_qnn::DeepInput input;
  input.start_sec = end_sec - 6.0;
  input.end_sec = end_sec;
  input.source_fps = 30.0;
  input.max_frame_gap_sec = 1.0 / 30.0;
  input.shape = {180, 1, 1, 3};
  input.tensor.reserve(180 * 3);
  for (int index = 0; index < 180; ++index) {
    const float green = 128.0F + 20.0F * static_cast<float>(std::sin(index * 0.2));
    input.tensor.insert(input.tensor.end(), {10.0F, green, 20.0F});
  }
  return input;
}

bool wait_for_result(rppg_qnn::DeepWorker& worker, double minimum_end_sec) {
  for (int attempt = 0; attempt < 100; ++attempt) {
    const auto result = worker.latest_result();
    if (result.has_value() && result->window_end_sec >= minimum_end_sec) {
      return true;
    }
    std::this_thread::sleep_for(20ms);
  }
  return false;
}

class ThrowingRuntime final : public rppg_qnn::IDeepRuntime {
 public:
  std::string backend_name() const override { return "throwing"; }
  rppg_qnn::HeartRateResult infer(const rppg_qnn::DeepInput&) override {
    throw std::runtime_error("inference failed");
  }
};

void fake_runtime_validates_and_returns_finite_result() {
  auto runtime = rppg_qnn::make_fake_deep_runtime(0ms);
  auto result = runtime->infer(valid_input(6.0));
  EXPECT_TRUE(result.is_valid);
  EXPECT_EQ(result.method, std::string("FAKE_DEEP"));
  EXPECT_EQ(result.backend, std::string("fake"));
  EXPECT_TRUE(std::isfinite(result.bpm));
  EXPECT_TRUE(std::isfinite(result.confidence));
  EXPECT_TRUE(std::isfinite(result.inference_ms));

  auto malformed = valid_input(6.0);
  malformed.shape = {180, 1, 1};
  result = runtime->infer(malformed);
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason, std::string("model_input_invalid"));
  EXPECT_TRUE(std::isfinite(result.window_start_sec));
  EXPECT_TRUE(std::isfinite(result.window_end_sec));
}

void worker_is_latest_only_nonblocking_and_closes_safely() {
  rppg_qnn::DeepWorker worker(rppg_qnn::make_fake_deep_runtime(100ms));
  const auto start = std::chrono::steady_clock::now();
  for (int index = 0; index < 20; ++index) {
    const auto submit_start = std::chrono::steady_clock::now();
    EXPECT_TRUE(worker.submit(valid_input(6.0 + index)));
    const auto submit_elapsed = std::chrono::steady_clock::now() - submit_start;
    EXPECT_TRUE(submit_elapsed < 5ms);
  }
  EXPECT_TRUE(std::chrono::steady_clock::now() - start < 20ms);
  EXPECT_TRUE(wait_for_result(worker, 25.0));
  const auto result = worker.latest_result();
  EXPECT_TRUE(result.has_value());
  if (result.has_value()) {
    EXPECT_EQ(result->window_end_sec, 25.0);
  }
  worker.close();
  worker.close();
  EXPECT_TRUE(!worker.submit(valid_input(30.0)));
}

void worker_converts_runtime_exceptions_into_safe_invalid_results() {
  rppg_qnn::DeepWorker worker(std::make_unique<ThrowingRuntime>());
  EXPECT_TRUE(worker.submit(valid_input(7.0)));
  EXPECT_TRUE(wait_for_result(worker, 7.0));
  const auto result = worker.latest_result();
  EXPECT_TRUE(result.has_value());
  if (result.has_value()) {
    EXPECT_TRUE(!result->is_valid);
    EXPECT_EQ(result->invalid_reason, std::string("model_inference_failed"));
    EXPECT_TRUE(std::isfinite(result->inference_ms));
  }
}

}  // namespace

int main() {
  fake_runtime_validates_and_returns_finite_result();
  worker_is_latest_only_nonblocking_and_closes_safely();
  worker_converts_runtime_exceptions_into_safe_invalid_results();
  return test_support::finish();
}
