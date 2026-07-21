#include "rppg_qnn/deep_worker.hpp"

#include "test_support.hpp"

#include <chrono>
#include <cmath>
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
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
  input.source_frame_count = 181;
  input.max_frame_gap_sec = 1.0 / 30.0;
  input.shape = {180, 1, 1, 3};
  input.tensor.reserve(180 * 3);
  for (int index = 0; index < 180; ++index) {
    const float green = 128.0F + 20.0F * static_cast<float>(
        std::sin(2.0 * 3.14159265358979323846 * 1.2 * index / 30.0));
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

struct BlockingState {
  std::mutex mutex;
  std::condition_variable condition;
  bool first_started{false};
  bool release{false};
  int calls{0};
};

class BlockingRuntime final : public rppg_qnn::IDeepRuntime {
 public:
  explicit BlockingRuntime(std::shared_ptr<BlockingState> state) : state_(std::move(state)) {}

  std::string backend_name() const override { return "blocking"; }

  rppg_qnn::HeartRateResult infer(const rppg_qnn::DeepInput& input) override {
    std::unique_lock<std::mutex> lock(state_->mutex);
    ++state_->calls;
    if (state_->calls == 1) {
      state_->first_started = true;
      state_->condition.notify_all();
      state_->condition.wait(lock, [this] { return state_->release; });
    }
    rppg_qnn::HeartRateResult result;
    result.method = "BLOCKING";
    result.backend = "blocking";
    result.window_start_sec = input.start_sec;
    result.window_end_sec = input.end_sec;
    result.source_fps = input.source_fps;
    result.source_frame_count = input.source_frame_count;
    result.max_frame_gap_sec = input.max_frame_gap_sec;
    result.is_valid = true;
    return result;
  }

 private:
  std::shared_ptr<BlockingState> state_;
};

void fake_runtime_estimates_sine_and_preserves_metadata() {
  auto runtime = rppg_qnn::make_fake_deep_runtime(20ms);
  const auto started = std::chrono::steady_clock::now();
  const auto result = runtime->infer(valid_input(6.0));
  const auto elapsed = std::chrono::steady_clock::now() - started;
  EXPECT_TRUE(result.is_valid);
  EXPECT_EQ(result.method, std::string("FAKE_DEEP"));
  EXPECT_EQ(result.backend, std::string("fake"));
  EXPECT_TRUE(std::abs(result.bpm - 72.0) <= 3.0);
  EXPECT_EQ(result.window_start_sec, 0.0);
  EXPECT_EQ(result.window_end_sec, 6.0);
  EXPECT_EQ(result.source_fps, 30.0);
  EXPECT_EQ(result.source_frame_count, 181U);
  EXPECT_EQ(result.max_frame_gap_sec, 1.0 / 30.0);
  EXPECT_TRUE(result.inference_ms >= 15.0);
  EXPECT_TRUE(elapsed >= 15ms);
  EXPECT_EQ(result.waveform.size(), 180U);
  bool non_flat = false;
  for (float value : result.waveform) {
    EXPECT_TRUE(std::isfinite(value));
    EXPECT_TRUE(value >= -1.0F && value <= 1.0F);
    non_flat = non_flat || std::abs(value) > 0.01F;
  }
  EXPECT_TRUE(non_flat);

  auto malformed = valid_input(6.0);
  malformed.shape = {180, 1, 1};
  const auto invalid = runtime->infer(malformed);
  EXPECT_TRUE(!invalid.is_valid);
  EXPECT_EQ(invalid.invalid_reason, std::string("model_input_invalid"));
  EXPECT_TRUE(std::isfinite(invalid.window_start_sec));
  EXPECT_TRUE(std::isfinite(invalid.window_end_sec));
}

void worker_is_latest_only_nonblocking_and_closes_safely() {
  rppg_qnn::DeepWorker worker(rppg_qnn::make_fake_deep_runtime(100ms));
  const auto start = std::chrono::steady_clock::now();
  for (int index = 0; index < 20; ++index) {
    const auto submit_start = std::chrono::steady_clock::now();
    EXPECT_TRUE(worker.submit(valid_input(6.0 + index)));
    EXPECT_TRUE(std::chrono::steady_clock::now() - submit_start < 5ms);
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

void close_drains_the_latest_pending_input() {
  const auto state = std::make_shared<BlockingState>();
  rppg_qnn::DeepWorker worker(std::make_unique<BlockingRuntime>(state));
  EXPECT_TRUE(worker.submit(valid_input(7.0)));
  {
    std::unique_lock<std::mutex> lock(state->mutex);
    EXPECT_TRUE(state->condition.wait_for(lock, 1s,
                                          [&] { return state->first_started; }));
  }
  EXPECT_TRUE(worker.submit(valid_input(8.0)));
  EXPECT_TRUE(worker.submit(valid_input(9.0)));
  std::atomic<bool> close_returned{false};
  std::thread closer([&worker, &close_returned] {
    worker.close();
    close_returned.store(true);
  });
  std::this_thread::sleep_for(25ms);
  EXPECT_TRUE(!close_returned.load());
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    state->release = true;
  }
  state->condition.notify_all();
  closer.join();
  EXPECT_TRUE(close_returned.load());
  {
    std::lock_guard<std::mutex> lock(state->mutex);
    EXPECT_EQ(state->calls, 2);
  }
  const auto result = worker.latest_result();
  EXPECT_TRUE(result.has_value());
  if (result.has_value()) {
    EXPECT_EQ(result->window_end_sec, 9.0);
  }
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
  fake_runtime_estimates_sine_and_preserves_metadata();
  worker_is_latest_only_nonblocking_and_closes_safely();
  close_drains_the_latest_pending_input();
  worker_converts_runtime_exceptions_into_safe_invalid_results();
  return test_support::finish();
}
