#include "rppg_qnn/tscan_runtime.hpp"

#include "rppg_qnn/error.hpp"
#include "rppg_qnn/tscan_postprocessor.hpp"
#include "rppg_qnn/tscan_preprocessor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace rppg_qnn {
namespace {

[[noreturn]] void fail(const char* message) {
  throw AppError(ErrorCode::InferenceFailed, message);
}

class TscanRuntime final : public IDeepRuntime {
 public:
  TscanRuntime(std::unique_ptr<ITscanSession> session, std::string backend)
      : session_(std::move(session)), backend_(std::move(backend)) {}

  [[nodiscard]] std::string backend_name() const override { return backend_; }

  HeartRateResult infer(const DeepInput& input) override {
    HeartRateResult result;
    result.window_materialization_ms =
        std::isfinite(input.window_materialization_ms) &&
                input.window_materialization_ms >= 0.0
            ? input.window_materialization_ms
            : 0.0;
    const auto preprocess_started = std::chrono::steady_clock::now();
    TscanTensor tensor = preprocess_tscan_rgb(input);
    result.preprocess_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - preprocess_started)
            .count();
    TscanModelOutput output = session_->run(tensor.values, tensor.shape);
    result.runtime_ms =
        std::isfinite(output.runtime_ms) && output.runtime_ms >= 0.0
            ? output.runtime_ms
            : 0.0;
    const auto postprocess_started = std::chrono::steady_clock::now();
    if (output.shape != std::vector<std::int64_t>{180, 1} ||
        output.waveform.size() != 180U) {
      fail("TSCAN_OUTPUT_SHAPE expected float32 [180,1]");
    }
    if (!std::all_of(output.waveform.begin(), output.waveform.end(),
                     [](float value) { return std::isfinite(value); })) {
      fail("TSCAN_OUTPUT_NONFINITE output contains nonfinite value");
    }

    const TscanPostprocessResult post =
        postprocess_tscan_waveform(output.waveform);
    result.method = "TSCAN";
    result.backend = backend_;
    result.window_start_sec = input.start_sec;
    result.window_end_sec = input.end_sec;
    result.source_fps = input.source_fps;
    result.source_frame_count = input.source_frame_count;
    result.max_frame_gap_sec = input.max_frame_gap_sec;
    result.bpm = post.bpm;
    result.confidence = post.confidence;
    result.is_valid = post.is_valid;
    result.invalid_reason = post.invalid_reason;
    result.waveform = std::move(output.waveform);
    result.postprocess_ms =
        std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - postprocess_started)
            .count();
    result.inference_ms =
        result.preprocess_ms + result.runtime_ms + result.postprocess_ms;
    return result;
  }

 private:
  std::unique_ptr<ITscanSession> session_;
  std::string backend_;
};

}  // namespace

std::unique_ptr<IDeepRuntime> make_tscan_runtime(
    std::unique_ptr<ITscanSession> session) {
  if (!session) {
    fail("TSCAN_SESSION_NULL session must not be null");
  }
  std::string backend = session->backend_name();
  if (backend.empty()) {
    fail("TSCAN_SESSION_BACKEND backend must not be empty");
  }
  return std::make_unique<TscanRuntime>(std::move(session), std::move(backend));
}

}  // namespace rppg_qnn
