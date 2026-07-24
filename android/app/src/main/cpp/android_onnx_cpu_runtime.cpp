#include "android_onnx_cpu_runtime.hpp"

#include "rppg_qnn/error.hpp"

#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rppg_qnn::android {
namespace {

constexpr std::size_t kSourceFrames = 180U;
constexpr std::size_t kModelFrames = 181U;
constexpr std::size_t kChannels = 3U;
constexpr std::size_t kHeight = 72U;
constexpr std::size_t kWidth = 72U;
constexpr double kSampleRateHz = 30.0;
constexpr double kMinimumHz = 0.7;
constexpr double kMaximumHz = 3.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kConfidenceThreshold = 0.10;
constexpr double kEpsilon = 1e-12;

HeartRateResult make_result(const DeepInput& input,
                            std::vector<float> waveform,
                            double inference_ms) {
  HeartRateResult result;
  result.method = "EFFICIENTPHYS";
  result.backend = "ONNX_RUNTIME_CPU";
  result.window_start_sec = input.start_sec;
  result.window_end_sec = input.end_sec;
  result.source_fps = input.source_fps;
  result.source_frame_count = input.source_frame_count;
  result.max_frame_gap_sec = input.max_frame_gap_sec;
  result.waveform = std::move(waveform);
  result.inference_ms = inference_ms;
  result.invalid_reason = "low_confidence";

  const std::size_t frames = result.waveform.size();
  const int first_bin = std::max(
      1, static_cast<int>(
             std::ceil(kMinimumHz * static_cast<double>(frames) / kSampleRateHz)));
  const int last_bin = std::min(
      static_cast<int>(frames / 2U),
      static_cast<int>(
          std::floor(kMaximumHz * static_cast<double>(frames) / kSampleRateHz)));
  double total_power = 0.0;
  double peak_power = 0.0;
  int peak_bin = first_bin;
  for (int bin = 1; bin <= static_cast<int>(frames / 2U); ++bin) {
    double real = 0.0;
    double imaginary = 0.0;
    for (std::size_t index = 0; index < frames; ++index) {
      const double phase =
          2.0 * kPi * static_cast<double>(bin) * static_cast<double>(index) /
          static_cast<double>(frames);
      real += static_cast<double>(result.waveform[index]) * std::cos(phase);
      imaginary -=
          static_cast<double>(result.waveform[index]) * std::sin(phase);
    }
    const double power = real * real + imaginary * imaginary;
    if (!std::isfinite(power)) {
      result.invalid_reason = "model_output_invalid";
      return result;
    }
    total_power += power;
    if (bin >= first_bin && bin <= last_bin && power > peak_power) {
      peak_power = power;
      peak_bin = bin;
    }
  }
  if (peak_power <= kEpsilon || total_power <= kEpsilon) {
    return result;
  }
  result.confidence = peak_power / total_power;
  if (std::isfinite(result.confidence) &&
      result.confidence >= kConfidenceThreshold) {
    result.is_valid = true;
    result.invalid_reason.clear();
    result.bpm = static_cast<double>(peak_bin) * kSampleRateHz * 60.0 /
                 static_cast<double>(frames);
  }
  return result;
}

std::vector<float> preprocess(const DeepInput& input) {
  const std::vector<std::int64_t> expected_shape{
      static_cast<std::int64_t>(kSourceFrames),
      static_cast<std::int64_t>(kHeight),
      static_cast<std::int64_t>(kWidth),
      static_cast<std::int64_t>(kChannels)};
  const std::size_t source_values =
      kSourceFrames * kHeight * kWidth * kChannels;
  if (input.shape != expected_shape || input.tensor.size() != source_values ||
      !std::all_of(input.tensor.begin(), input.tensor.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw AppError(ErrorCode::InferenceFailed,
                   "EfficientPhys requires finite float32 [180,72,72,3] RGB");
  }

  const double mean =
      std::accumulate(input.tensor.begin(), input.tensor.end(), 0.0) /
      static_cast<double>(input.tensor.size());
  double variance = 0.0;
  for (float value : input.tensor) {
    const double centered = static_cast<double>(value) - mean;
    variance += centered * centered;
  }
  variance /= static_cast<double>(input.tensor.size());
  const double standard_deviation = std::sqrt(variance);
  if (!std::isfinite(standard_deviation) || standard_deviation <= kEpsilon) {
    throw AppError(ErrorCode::InferenceFailed,
                   "EfficientPhys input standard deviation is zero");
  }

  std::vector<float> model_input(kModelFrames * kChannels * kHeight * kWidth);
  for (std::size_t frame = 0; frame < kModelFrames; ++frame) {
    const std::size_t source_frame = std::min(frame, kSourceFrames - 1U);
    for (std::size_t channel = 0; channel < kChannels; ++channel) {
      for (std::size_t row = 0; row < kHeight; ++row) {
        for (std::size_t column = 0; column < kWidth; ++column) {
          const std::size_t source_index =
              ((source_frame * kHeight + row) * kWidth + column) * kChannels +
              channel;
          const std::size_t destination_index =
              ((frame * kChannels + channel) * kHeight + row) * kWidth + column;
          model_input[destination_index] = static_cast<float>(
              (static_cast<double>(input.tensor[source_index]) - mean) /
              standard_deviation);
        }
      }
    }
  }
  return model_input;
}

class OnnxCpuRuntime final : public IDeepRuntime {
 public:
  explicit OnnxCpuRuntime(const std::string& model_path)
      : environment_(ORT_LOGGING_LEVEL_WARNING, "rppg-efficientphys"),
        session_options_(),
        session_(nullptr) {
    if (!std::filesystem::is_regular_file(model_path)) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     "EfficientPhys ONNX model is missing: " + model_path);
    }
    session_options_.SetIntraOpNumThreads(2);
    session_options_.SetInterOpNumThreads(1);
    session_options_.SetGraphOptimizationLevel(
        GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = Ort::Session(environment_, model_path.c_str(), session_options_);
  }

  [[nodiscard]] std::string backend_name() const override {
    return "ONNX_RUNTIME_CPU";
  }

  HeartRateResult infer(const DeepInput& input) override {
    const auto started = std::chrono::steady_clock::now();
    try {
      std::vector<float> model_input = preprocess(input);
      const std::array<std::int64_t, 4> input_shape{
          static_cast<std::int64_t>(kModelFrames),
          static_cast<std::int64_t>(kChannels),
          static_cast<std::int64_t>(kHeight),
          static_cast<std::int64_t>(kWidth)};
      Ort::MemoryInfo memory =
          Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory, model_input.data(), model_input.size(), input_shape.data(),
          input_shape.size());
      constexpr const char* input_names[] = {"frames"};
      constexpr const char* output_names[] = {"pulse"};
      std::vector<Ort::Value> outputs =
          session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1U,
                       output_names, 1U);
      if (outputs.size() != 1U || !outputs.front().IsTensor()) {
        throw AppError(ErrorCode::InferenceFailed,
                       "EfficientPhys did not return one pulse tensor");
      }
      const Ort::TensorTypeAndShapeInfo info =
          outputs.front().GetTensorTypeAndShapeInfo();
      if (info.GetElementType() !=
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          info.GetShape() !=
              std::vector<std::int64_t>{
                  static_cast<std::int64_t>(kSourceFrames), 1}) {
        throw AppError(ErrorCode::InferenceFailed,
                       "EfficientPhys output must be float32 [180,1]");
      }
      const float* output = outputs.front().GetTensorData<float>();
      std::vector<float> waveform(output, output + kSourceFrames);
      if (!std::all_of(waveform.begin(), waveform.end(),
                       [](float value) { return std::isfinite(value); })) {
        throw AppError(ErrorCode::InferenceFailed,
                       "EfficientPhys output contains non-finite values");
      }
      const double inference_ms =
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started)
              .count();
      return make_result(input, std::move(waveform), inference_ms);
    } catch (const AppError&) {
      throw;
    } catch (const Ort::Exception& error) {
      throw AppError(ErrorCode::InferenceFailed,
                     std::string("ONNX Runtime CPU failed: ") + error.what());
    }
  }

 private:
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  Ort::Session session_;
};

}  // namespace

std::unique_ptr<IDeepRuntime> make_onnx_cpu_runtime(
    const std::string& model_path) {
  return std::make_unique<OnnxCpuRuntime>(model_path);
}

}  // namespace rppg_qnn::android
