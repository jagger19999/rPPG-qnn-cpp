#include "rppg_qnn/efficientphys_runtime.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kSourceValues = 180U * 72U * 72U * 3U;
constexpr std::size_t kModelValues = 181U * 3U * 72U * 72U;

rppg_qnn::DeepInput valid_input() {
  rppg_qnn::DeepInput input;
  input.start_sec = 10.0;
  input.end_sec = 16.0;
  input.source_fps = 30.0;
  input.max_frame_gap_sec = 1.0 / 30.0;
  input.source_frame_count = 181U;
  input.shape = {180, 72, 72, 3};
  input.tensor.resize(kSourceValues);
  for (std::size_t index = 0; index < input.tensor.size(); ++index) {
    input.tensor[index] = index % 2U == 0U ? 0.0F : 2.0F;
  }
  return input;
}

struct SessionState {
  int calls{0};
  std::vector<std::int64_t> input_shape;
  std::vector<float> input;
  std::vector<std::int64_t> output_shape{180, 1};
  std::size_t output_count{180U};
  bool constant_output{false};
  bool nonfinite_output{false};
};

class RecordingSession final : public rppg_qnn::IEfficientPhysSession {
 public:
  explicit RecordingSession(std::shared_ptr<SessionState> state,
                            std::string backend = "onnxruntime_cpu")
      : state_(std::move(state)), backend_(std::move(backend)) {}

  std::string backend_name() const override { return backend_; }

  rppg_qnn::EfficientPhysModelOutput run(
      const std::vector<float>& input,
      const std::vector<std::int64_t>& shape) override {
    ++state_->calls;
    state_->input = input;
    state_->input_shape = shape;
    std::vector<float> output(state_->output_count);
    for (std::size_t index = 0; index < output.size(); ++index) {
      output[index] = state_->constant_output
                          ? 1.0F
                          : static_cast<float>(
                                std::sin(2.0 * 3.14159265358979323846 * 1.2 *
                                         static_cast<double>(index) / 30.0));
    }
    if (state_->nonfinite_output) {
      output[7] = std::numeric_limits<float>::quiet_NaN();
    }
    return {std::move(output), state_->output_shape};
  }

 private:
  std::shared_ptr<SessionState> state_;
  std::string backend_;
};

void exact_preprocessing_and_result_contract() {
  auto state = std::make_shared<SessionState>();
  auto runtime = rppg_qnn::make_onnxruntime_efficientphys_runtime(
      std::make_unique<RecordingSession>(state));
  EXPECT_EQ(runtime->backend_name(), std::string("onnxruntime_cpu"));

  const auto result = runtime->infer(valid_input());
  EXPECT_EQ(state->calls, 1);
  EXPECT_EQ(state->input_shape,
            (std::vector<std::int64_t>{181, 3, 72, 72}));
  EXPECT_EQ(state->input.size(), kModelValues);
  EXPECT_TRUE(std::abs(state->input[0] - -1.0F) < 1e-6F);
  EXPECT_TRUE(std::abs(state->input[1] - 1.0F) < 1e-6F);
  EXPECT_TRUE(std::abs(state->input[72U * 72U] - 1.0F) < 1e-6F);
  const std::size_t last_source_frame = 179U * 3U * 72U * 72U;
  const std::size_t appended_frame = 180U * 3U * 72U * 72U;
  EXPECT_EQ(state->input[appended_frame], state->input[last_source_frame]);
  EXPECT_TRUE(std::equal(state->input.begin() + last_source_frame,
                         state->input.begin() + appended_frame,
                         state->input.begin() + appended_frame));

  EXPECT_EQ(result.method, std::string("EFFICIENTPHYS"));
  EXPECT_EQ(result.backend, std::string("onnxruntime_cpu"));
  EXPECT_TRUE(result.is_valid);
  EXPECT_TRUE(std::abs(result.bpm - 72.0) <= 0.01);
  EXPECT_EQ(result.waveform.size(), 180U);
  EXPECT_EQ(result.window_start_sec, 10.0);
  EXPECT_EQ(result.window_end_sec, 16.0);
  EXPECT_EQ(result.source_frame_count, 181U);
  EXPECT_TRUE(std::isfinite(result.confidence));
  EXPECT_TRUE(std::isfinite(result.inference_ms));
}

void malformed_input_never_reaches_session() {
  auto state = std::make_shared<SessionState>();
  auto runtime = rppg_qnn::make_onnxruntime_efficientphys_runtime(
      std::make_unique<RecordingSession>(state));

  const auto expect_rejected = [&](const rppg_qnn::DeepInput& input) {
    const int calls_before = state->calls;
    const auto result = runtime->infer(input);
    EXPECT_EQ(state->calls, calls_before);
    EXPECT_TRUE(!result.is_valid);
    EXPECT_EQ(result.invalid_reason,
              std::string("efficientphys_input_invalid"));
    EXPECT_EQ(result.backend, std::string("onnxruntime_cpu"));
    EXPECT_EQ(result.method, std::string("EFFICIENTPHYS"));
    EXPECT_TRUE(std::isfinite(result.inference_ms));
  };

  auto wrong_frames = valid_input();
  wrong_frames.shape = {179, 72, 72, 3};
  expect_rejected(wrong_frames);

  auto wrong_height = valid_input();
  wrong_height.shape = {180, 71, 72, 3};
  expect_rejected(wrong_height);

  auto wrong_width = valid_input();
  wrong_width.shape = {180, 72, 71, 3};
  expect_rejected(wrong_width);

  auto wrong_channels = valid_input();
  wrong_channels.shape = {180, 72, 72, 1};
  expect_rejected(wrong_channels);

  auto wrong_length = valid_input();
  wrong_length.tensor.pop_back();
  expect_rejected(wrong_length);

  auto nonfinite = valid_input();
  nonfinite.tensor[7] = std::numeric_limits<float>::infinity();
  expect_rejected(nonfinite);
}

void zero_variance_input_matches_reference() {
  auto state = std::make_shared<SessionState>();
  auto runtime = rppg_qnn::make_onnxruntime_efficientphys_runtime(
      std::make_unique<RecordingSession>(state));
  auto degenerate = valid_input();
  std::fill(degenerate.tensor.begin(), degenerate.tensor.end(), 4.0F);
  const auto result = runtime->infer(degenerate);
  EXPECT_EQ(state->calls, 1);
  EXPECT_EQ(state->input_shape,
            (std::vector<std::int64_t>{181, 3, 72, 72}));
  EXPECT_EQ(state->input.size(), kModelValues);
  EXPECT_TRUE(std::all_of(state->input.begin(), state->input.end(),
                          [](float value) {
                            return std::isfinite(value) && value == 0.0F;
                          }));
  EXPECT_TRUE(result.is_valid);
}

void invalid_model_output_is_concrete_and_finite() {
  auto state = std::make_shared<SessionState>();
  state->output_shape = {180};
  auto runtime = rppg_qnn::make_onnxruntime_efficientphys_runtime(
      std::make_unique<RecordingSession>(state));
  auto result = runtime->infer(valid_input());
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason,
            std::string("efficientphys_output_invalid"));
  EXPECT_TRUE(std::isfinite(result.inference_ms));

  state = std::make_shared<SessionState>();
  state->output_count = 179U;
  runtime = rppg_qnn::make_onnxruntime_efficientphys_runtime(
      std::make_unique<RecordingSession>(state));
  result = runtime->infer(valid_input());
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason,
            std::string("efficientphys_output_invalid"));

  state = std::make_shared<SessionState>();
  state->nonfinite_output = true;
  runtime = rppg_qnn::make_onnxruntime_efficientphys_runtime(
      std::make_unique<RecordingSession>(state));
  result = runtime->infer(valid_input());
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason,
            std::string("efficientphys_output_invalid"));

  state = std::make_shared<SessionState>();
  state->constant_output = true;
  runtime = rppg_qnn::make_onnxruntime_efficientphys_runtime(
      std::make_unique<RecordingSession>(state));
  result = runtime->infer(valid_input());
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.bpm, 0.0);
  EXPECT_EQ(result.invalid_reason, std::string("TSCAN_WAVEFORM_INVALID"));
  EXPECT_TRUE(std::isfinite(result.confidence));
}

void backend_mismatch_is_rejected_without_fallback() {
  bool threw = false;
  try {
    auto state = std::make_shared<SessionState>();
    (void)rppg_qnn::make_onnxruntime_efficientphys_runtime(
        std::make_unique<RecordingSession>(state, "fake"));
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}

}  // namespace

int main() {
  exact_preprocessing_and_result_contract();
  malformed_input_never_reaches_session();
  zero_variance_input_matches_reference();
  invalid_model_output_is_concrete_and_finite();
  backend_mismatch_is_rejected_without_fallback();
  return test_support::finish();
}
