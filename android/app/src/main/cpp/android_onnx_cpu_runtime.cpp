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
constexpr std::size_t kModelFrames = 180U;
constexpr std::size_t kInChannels = 6U;
constexpr std::size_t kRgbChannels = 3U;
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
  result.method = "TSCAN";
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
  // Input is [180, 72, 72, 3] NHWC RGB.
  const std::vector<std::int64_t> expected_shape{
      static_cast<std::int64_t>(kSourceFrames),
      static_cast<std::int64_t>(kHeight),
      static_cast<std::int64_t>(kWidth),
      static_cast<std::int64_t>(kRgbChannels)};
  const std::size_t source_values =
      kSourceFrames * kHeight * kWidth * kRgbChannels;
  if (input.shape != expected_shape || input.tensor.size() != source_values ||
      !std::all_of(input.tensor.begin(), input.tensor.end(),
                   [](float value) { return std::isfinite(value); })) {
    throw AppError(ErrorCode::InferenceFailed,
                   "TSCAN requires finite float32 [180,72,72,3] RGB");
  }

  const std::size_t pixels_per_frame = kHeight * kWidth;
  const std::size_t frame_stride = pixels_per_frame * kRgbChannels;

  // --- Compute diff frames: diff[t] = frame[t] - frame[t-1], diff[0] = 0 ---
  // Also compute per-frame channel means for the first-frame diff fill.
  std::vector<double> diff_data(source_values, 0.0);
  double diff_sum = 0.0;
  for (std::size_t f = 1; f < kSourceFrames; ++f) {
    for (std::size_t p = 0; p < pixels_per_frame; ++p) {
      for (std::size_t c = 0; c < kRgbChannels; ++c) {
        const std::size_t idx_curr = f * frame_stride + p * kRgbChannels + c;
        const std::size_t idx_prev = (f - 1) * frame_stride + p * kRgbChannels + c;
        const double d = static_cast<double>(input.tensor[idx_curr]) -
                         static_cast<double>(input.tensor[idx_prev]);
        diff_data[idx_curr] = d;
        diff_sum += d;
      }
    }
  }
  // Fill frame 0 diff with the mean diff value to avoid all-zeros.
  const double diff_mean =
      diff_sum / static_cast<double>((kSourceFrames - 1) * pixels_per_frame * kRgbChannels);
  for (std::size_t p = 0; p < pixels_per_frame; ++p) {
    for (std::size_t c = 0; c < kRgbChannels; ++c) {
      diff_data[p * kRgbChannels + c] = diff_mean;
    }
  }

  // --- Standardize diff frames ---
  double diff_var = 0.0;
  for (double v : diff_data) {
    const double centered = v - diff_mean;
    diff_var += centered * centered;
  }
  diff_var /= static_cast<double>(diff_data.size());
  const double diff_std = std::sqrt(diff_var);
  if (!std::isfinite(diff_std) || diff_std <= kEpsilon) {
    throw AppError(ErrorCode::InferenceFailed,
                   "TSCAN diff standard deviation is zero");
  }

  // --- Standardize raw frames ---
  double raw_sum = 0.0;
  for (float v : input.tensor) {
    raw_sum += static_cast<double>(v);
  }
  const double raw_mean = raw_sum / static_cast<double>(input.tensor.size());
  double raw_var = 0.0;
  for (float v : input.tensor) {
    const double centered = static_cast<double>(v) - raw_mean;
    raw_var += centered * centered;
  }
  raw_var /= static_cast<double>(input.tensor.size());
  const double raw_std = std::sqrt(raw_var);
  if (!std::isfinite(raw_std) || raw_std <= kEpsilon) {
    throw AppError(ErrorCode::InferenceFailed,
                   "TSCAN raw standard deviation is zero");
  }

  // --- Pack into NCHW [180, 6, 72, 72] ---
  // Channel layout: [diff_R, diff_G, diff_B, raw_R, raw_G, raw_B]
  std::vector<float> model_input(kModelFrames * kInChannels * kHeight * kWidth, 0.0f);
  for (std::size_t f = 0; f < kModelFrames; ++f) {
    for (std::size_t row = 0; row < kHeight; ++row) {
      for (std::size_t col = 0; col < kWidth; ++col) {
        const std::size_t px = row * kWidth + col;
        const std::size_t fp = f * pixels_per_frame + px;
        const std::size_t base =
            (f * kInChannels * kHeight + row) * kWidth + col;

        // Channels 0-2: diff-normalized RGB
        for (std::size_t c = 0; c < kRgbChannels; ++c) {
          const std::size_t src = fp * kRgbChannels + c;
          const std::size_t dst = base + c * kHeight * kWidth;
          model_input[dst] = static_cast<float>(
              (diff_data[src] - diff_mean) / diff_std);
        }

        // Channels 3-5: standardized raw RGB
        for (std::size_t c = 0; c < kRgbChannels; ++c) {
          const std::size_t src = f * frame_stride + px * kRgbChannels + c;
          const std::size_t dst = base + (kRgbChannels + c) * kHeight * kWidth;
          model_input[dst] = static_cast<float>(
              (static_cast<double>(input.tensor[src]) - raw_mean) / raw_std);
        }
      }
    }
  }
  return model_input;
}

class OnnxCpuRuntime final : public IDeepRuntime {
 public:
  explicit OnnxCpuRuntime(const std::string& model_path)
      : environment_(ORT_LOGGING_LEVEL_WARNING, "rppg-tscan"),
        session_options_(),
        session_(nullptr) {
    if (!std::filesystem::is_regular_file(model_path)) {
      throw AppError(ErrorCode::ModelLoadFailed,
                     "TSCAN ONNX model is missing: " + model_path);
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
          static_cast<std::int64_t>(kInChannels),
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
                       "TSCAN did not return one pulse tensor");
      }
      const Ort::TensorTypeAndShapeInfo info =
          outputs.front().GetTensorTypeAndShapeInfo();
      if (info.GetElementType() !=
              ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          info.GetShape() !=
              std::vector<std::int64_t>{
                  static_cast<std::int64_t>(kSourceFrames), 1}) {
        throw AppError(ErrorCode::InferenceFailed,
                       "TSCAN output must be float32 [180,1]");
      }
      const float* output = outputs.front().GetTensorData<float>();
      std::vector<float> waveform(output, output + kSourceFrames);
      if (!std::all_of(waveform.begin(), waveform.end(),
                       [](float value) { return std::isfinite(value); })) {
        throw AppError(ErrorCode::InferenceFailed,
                       "TSCAN output contains non-finite values");
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
