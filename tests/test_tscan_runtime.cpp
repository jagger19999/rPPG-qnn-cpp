#include "rppg_qnn/tscan_runtime.hpp"

#include "rppg_qnn/error.hpp"
#include "rppg_qnn/tscan_preprocessor.hpp"
#include "test_support.hpp"

#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kInputValues = 180U * 72U * 72U * 3U;

rppg_qnn::DeepInput valid_input() {
  rppg_qnn::DeepInput input;
  input.start_sec = 10.0;
  input.end_sec = 16.0;
  input.source_fps = 30.0;
  input.max_frame_gap_sec = 0.04;
  input.source_frame_count = 180U;
  input.window_materialization_ms = 1.25;
  input.shape = {180, 72, 72, 3};
  input.tensor.resize(kInputValues);
  for (std::size_t index = 0; index < input.tensor.size(); ++index) {
    input.tensor[index] = static_cast<float>((index * 17U + index / 101U) % 251U);
  }
  return input;
}

struct SessionState {
  int calls{0};
  mutable int backend_calls{0};
  std::vector<float> input;
  std::vector<std::int64_t> shape;
  std::vector<std::int64_t> output_shape{180, 1};
  bool nonfinite{false};
  bool constant{false};
  std::chrono::milliseconds delay{0};
  double runtime_ms{0.0};
  bool throws{false};
};

class RecordingSession final : public rppg_qnn::ITscanSession {
 public:
  explicit RecordingSession(std::shared_ptr<SessionState> state,
                            std::string backend = "ONNX_RUNTIME_CPU")
      : state_(std::move(state)), backend_(std::move(backend)) {}

  std::string backend_name() const override {
    ++state_->backend_calls;
    return backend_;
  }

  rppg_qnn::TscanModelOutput run(
      const std::vector<float>& values,
      const std::vector<std::int64_t>& shape) override {
    ++state_->calls;
    std::this_thread::sleep_for(state_->delay);
    if (state_->throws) {
      throw std::runtime_error("session failed");
    }
    state_->input = values;
    state_->shape = shape;
    std::vector<float> waveform(180U);
    for (std::size_t index = 0; index < waveform.size(); ++index) {
      waveform[index] = state_->constant
                            ? 4.0F
                            : static_cast<float>(std::sin(
                                  2.0 * 3.14159265358979323846 * 1.0 *
                                  static_cast<double>(index) / 30.0));
    }
    if (state_->nonfinite) {
      waveform[12] = std::numeric_limits<float>::infinity();
    }
    return {std::move(waveform), state_->output_shape, state_->runtime_ms};
  }

 private:
  std::shared_ptr<SessionState> state_;
  std::string backend_;
};

template <typename Function>
void expect_inference_error(Function function, const std::string& token) {
  bool threw = false;
  try {
    function();
  } catch (const rppg_qnn::AppError& error) {
    threw = true;
    EXPECT_EQ(error.code(), rppg_qnn::ErrorCode::InferenceFailed);
    EXPECT_TRUE(std::string(error.what()).find(token) != std::string::npos);
  }
  EXPECT_TRUE(threw);
}

void composes_preprocess_session_and_postprocess() {
  const auto input = valid_input();
  const auto expected = rppg_qnn::preprocess_tscan_rgb(input);
  auto state = std::make_shared<SessionState>();
  state->delay = std::chrono::milliseconds(50);
  state->runtime_ms = 1.0;
  auto runtime = rppg_qnn::make_tscan_runtime(
      std::make_unique<RecordingSession>(state));

  const auto result = runtime->infer(input);

  EXPECT_EQ(state->calls, 1);
  EXPECT_EQ(state->backend_calls, 1);
  EXPECT_EQ(state->shape, (std::vector<std::int64_t>{180, 6, 72, 72}));
  EXPECT_EQ(state->input.size(), expected.values.size());
  for (std::size_t index : std::vector<std::size_t>{
           0U, 72U * 72U, 3U * 72U * 72U, expected.values.size() - 1U}) {
    EXPECT_TRUE(std::abs(state->input[index] - expected.values[index]) < 1e-6F);
  }
  EXPECT_EQ(result.method, std::string("TSCAN"));
  EXPECT_EQ(result.backend, std::string("ONNX_RUNTIME_CPU"));
  EXPECT_EQ(result.window_start_sec, input.start_sec);
  EXPECT_EQ(result.window_end_sec, input.end_sec);
  EXPECT_EQ(result.source_fps, input.source_fps);
  EXPECT_EQ(result.source_frame_count, input.source_frame_count);
  EXPECT_EQ(result.max_frame_gap_sec, input.max_frame_gap_sec);
  EXPECT_EQ(result.waveform.size(), 180U);
  EXPECT_TRUE(result.is_valid);
  EXPECT_TRUE(std::abs(result.bpm - 60.0) < 0.01);
  EXPECT_EQ(result.raw_bpm, result.bpm);
  EXPECT_EQ(result.display_bpm, result.raw_bpm);
  EXPECT_TRUE(result.stability_valid);
  EXPECT_EQ(result.correction_reason, std::string("accepted_raw"));
  EXPECT_TRUE(std::isfinite(result.confidence));
  EXPECT_TRUE(std::isfinite(result.inference_ms));
  EXPECT_EQ(result.window_materialization_ms, 1.25);
  EXPECT_TRUE(std::isfinite(result.preprocess_ms));
  EXPECT_TRUE(result.preprocess_ms >= 0.0);
  EXPECT_EQ(result.runtime_ms, 1.0);
  EXPECT_TRUE(std::isfinite(result.postprocess_ms));
  EXPECT_TRUE(result.postprocess_ms >= 0.0);
  const double stage_sum = result.preprocess_ms + result.runtime_ms +
                           result.postprocess_ms;
  EXPECT_TRUE(result.inference_ms >= 40.0);
  EXPECT_TRUE(result.inference_ms > stage_sum);
}

void session_exception_preserves_completed_timing() {
  auto state = std::make_shared<SessionState>();
  state->delay = std::chrono::milliseconds(50);
  state->runtime_ms = 99.0;
  state->throws = true;
  auto runtime = rppg_qnn::make_tscan_runtime(
      std::make_unique<RecordingSession>(state));

  const auto result = runtime->infer(valid_input());

  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason, std::string("deep_runtime_exception"));
  EXPECT_TRUE(result.preprocess_ms >= 0.0);
  EXPECT_EQ(result.runtime_ms, 0.0);
  EXPECT_EQ(result.postprocess_ms, 0.0);
  EXPECT_TRUE(result.inference_ms >= 40.0);
}

void malformed_input_never_reaches_session() {
  auto state = std::make_shared<SessionState>();
  auto runtime = rppg_qnn::make_tscan_runtime(
      std::make_unique<RecordingSession>(state));
  auto input = valid_input();
  input.shape = {180, 72, 72, 1};
  expect_inference_error([&] { (void)runtime->infer(input); },
                         "TSCAN_PREPROCESS_SHAPE");
  EXPECT_EQ(state->calls, 0);
}

void malformed_outputs_throw() {
  auto state = std::make_shared<SessionState>();
  state->output_shape = {180};
  auto runtime = rppg_qnn::make_tscan_runtime(
      std::make_unique<RecordingSession>(state));
  expect_inference_error([&] { (void)runtime->infer(valid_input()); },
                         "TSCAN_OUTPUT_SHAPE");

  state = std::make_shared<SessionState>();
  state->nonfinite = true;
  runtime = rppg_qnn::make_tscan_runtime(
      std::make_unique<RecordingSession>(state));
  expect_inference_error([&] { (void)runtime->infer(valid_input()); },
                         "TSCAN_OUTPUT_NONFINITE");
}

void invalid_sessions_are_rejected() {
  expect_inference_error(
      [] { (void)rppg_qnn::make_tscan_runtime(nullptr); },
      "TSCAN_SESSION_NULL");
  auto state = std::make_shared<SessionState>();
  expect_inference_error([&] {
    (void)rppg_qnn::make_tscan_runtime(
        std::make_unique<RecordingSession>(state, ""));
  }, "TSCAN_SESSION_BACKEND");
  EXPECT_EQ(state->backend_calls, 1);
}

void constant_output_returns_waveform_invalid_result() {
  auto state = std::make_shared<SessionState>();
  state->constant = true;
  auto runtime = rppg_qnn::make_tscan_runtime(
      std::make_unique<RecordingSession>(state));
  const auto result = runtime->infer(valid_input());
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason, std::string{"TSCAN_WAVEFORM_INVALID"});
  EXPECT_EQ(result.bpm, 0.0);
  EXPECT_EQ(result.raw_bpm, 0.0);
  EXPECT_EQ(result.display_bpm, 0.0);
  EXPECT_TRUE(!result.stability_valid);
  EXPECT_EQ(result.correction_reason,
            std::string("rejected_constant_waveform"));
  EXPECT_EQ(result.confidence, 0.0);
  EXPECT_EQ(result.waveform, std::vector<float>(180U, 4.0F));
}

}  // namespace

int main() {
  composes_preprocess_session_and_postprocess();
  session_exception_preserves_completed_timing();
  malformed_input_never_reaches_session();
  malformed_outputs_throw();
  invalid_sessions_are_rejected();
  constant_output_returns_waveform_invalid_result();
  return test_support::finish();
}
