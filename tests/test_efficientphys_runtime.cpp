#include "rppg_qnn/efficientphys_runtime.hpp"

#include "test_support.hpp"

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
    std::vector<float> output(180U);
    for (std::size_t index = 0; index < output.size(); ++index) {
      output[index] = static_cast<float>(
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
  auto input = valid_input();
  input.shape = {180, 72, 72, 1};
  const auto result = runtime->infer(input);
  EXPECT_EQ(state->calls, 0);
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason, std::string("efficientphys_input_invalid"));
  EXPECT_EQ(result.backend, std::string("onnxruntime_cpu"));
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
  state->nonfinite_output = true;
  runtime = rppg_qnn::make_onnxruntime_efficientphys_runtime(
      std::make_unique<RecordingSession>(state));
  result = runtime->infer(valid_input());
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason,
            std::string("efficientphys_output_invalid"));
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
  invalid_model_output_is_concrete_and_finite();
  backend_mismatch_is_rejected_without_fallback();
  return test_support::finish();
}
