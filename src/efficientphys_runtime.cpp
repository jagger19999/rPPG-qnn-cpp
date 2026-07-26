#include "rppg_qnn/efficientphys_runtime.hpp"

#include "rppg_qnn/tscan_postprocessor.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rppg_qnn {
namespace {

constexpr std::size_t kSourceFrames = 180U;
constexpr std::size_t kModelFrames = 181U;
constexpr std::size_t kHeight = 72U;
constexpr std::size_t kWidth = 72U;
constexpr std::size_t kChannels = 3U;
constexpr std::size_t kPixels = kHeight * kWidth;
constexpr std::size_t kFrameValues = kChannels * kPixels;
constexpr std::size_t kSourceValues = kSourceFrames * kFrameValues;
constexpr std::size_t kModelValues = kModelFrames * kFrameValues;
constexpr double kPi = 3.14159265358979323846;
constexpr double kOutputFps = 30.0;
constexpr double kMinimumFrequency = 0.75;
constexpr double kMaximumFrequency = 2.5;

using Clock = std::chrono::steady_clock;

std::size_t source_offset(std::size_t frame, std::size_t pixel,
                          std::size_t channel) {
  return (frame * kPixels + pixel) * kChannels + channel;
}

std::size_t model_offset(std::size_t frame, std::size_t channel,
                         std::size_t pixel) {
  return (frame * kChannels + channel) * kPixels + pixel;
}

HeartRateResult base_result(const DeepInput& input, const std::string& backend) {
  HeartRateResult result;
  result.method = "EFFICIENTPHYS";
  result.backend = backend;
  result.window_start_sec = input.start_sec;
  result.window_end_sec = input.end_sec;
  result.source_fps = input.source_fps;
  result.source_frame_count = input.source_frame_count;
  result.max_frame_gap_sec = input.max_frame_gap_sec;
  return result;
}

void set_elapsed(HeartRateResult* result, Clock::time_point started) {
  result->inference_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - started).count();
}

bool input_contract_is_valid(const DeepInput& input) {
  return input.shape == std::vector<std::int64_t>{180, 72, 72, 3} &&
         input.tensor.size() == kSourceValues &&
         std::all_of(input.tensor.begin(), input.tensor.end(),
                     [](float value) { return std::isfinite(value); });
}

double refined_bpm(const std::vector<float>& waveform, double coarse_bpm) {
  const double mean =
      std::accumulate(waveform.begin(), waveform.end(), 0.0) /
      static_cast<double>(waveform.size());
  std::vector<double> windowed(waveform.size());
  for (std::size_t sample = 0; sample < waveform.size(); ++sample) {
    const double window =
        0.5 - 0.5 * std::cos(2.0 * kPi * static_cast<double>(sample) /
                             static_cast<double>(waveform.size() - 1U));
    windowed[sample] =
        (static_cast<double>(waveform[sample]) - mean) * window;
  }

  const auto power_at = [&](double frequency) {
    double real = 0.0;
    double imaginary = 0.0;
    for (std::size_t sample = 0; sample < windowed.size(); ++sample) {
      const double angle =
          2.0 * kPi * frequency * static_cast<double>(sample) / kOutputFps;
      real += windowed[sample] * std::cos(angle);
      imaginary -= windowed[sample] * std::sin(angle);
    }
    return real * real + imaginary * imaginary;
  };

  const double bin_width =
      kOutputFps / static_cast<double>(waveform.size());
  const double coarse_frequency = coarse_bpm / 60.0;
  double lower =
      std::max(kMinimumFrequency, coarse_frequency - bin_width);
  double upper =
      std::min(kMaximumFrequency, coarse_frequency + bin_width);
  for (int iteration = 0; iteration < 48; ++iteration) {
    const double third = (upper - lower) / 3.0;
    const double left = lower + third;
    const double right = upper - third;
    if (power_at(left) < power_at(right)) {
      lower = left;
    } else {
      upper = right;
    }
  }
  return 30.0 * (lower + upper);
}

class EfficientPhysRuntime final : public IDeepRuntime {
 public:
  EfficientPhysRuntime(std::unique_ptr<IEfficientPhysSession> session,
                       std::string backend)
      : session_(std::move(session)), backend_(std::move(backend)) {}

  [[nodiscard]] std::string backend_name() const override { return backend_; }

  HeartRateResult infer(const DeepInput& input) override {
    const auto started = Clock::now();
    HeartRateResult result = base_result(input, backend_);
    if (!input_contract_is_valid(input)) {
      result.invalid_reason = "efficientphys_input_invalid";
      set_elapsed(&result, started);
      return result;
    }

    double sum = 0.0;
    for (float value : input.tensor) {
      sum += static_cast<double>(value);
    }
    const double mean = sum / static_cast<double>(input.tensor.size());
    double squared_deviations = 0.0;
    for (float value : input.tensor) {
      const double deviation = static_cast<double>(value) - mean;
      squared_deviations += deviation * deviation;
    }
    const double standard_deviation =
        std::sqrt(squared_deviations /
                  static_cast<double>(input.tensor.size()));
    if (!std::isfinite(standard_deviation) || standard_deviation == 0.0) {
      result.invalid_reason = "efficientphys_input_invalid";
      set_elapsed(&result, started);
      return result;
    }

    std::vector<float> model_input(kModelValues);
    for (std::size_t frame = 0; frame < kSourceFrames; ++frame) {
      for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
        for (std::size_t channel = 0; channel < kChannels; ++channel) {
          const double normalized =
              (static_cast<double>(
                   input.tensor[source_offset(frame, pixel, channel)]) -
               mean) /
              standard_deviation;
          model_input[model_offset(frame, channel, pixel)] =
              static_cast<float>(normalized);
        }
      }
    }
    std::copy_n(model_input.begin() +
                    static_cast<std::ptrdiff_t>((kSourceFrames - 1U) *
                                                kFrameValues),
                kFrameValues,
                model_input.begin() +
                    static_cast<std::ptrdiff_t>(kSourceFrames * kFrameValues));

    EfficientPhysModelOutput output =
        session_->run(model_input, {181, 3, 72, 72});
    if (output.shape != std::vector<std::int64_t>{180, 1} ||
        output.waveform.size() != kSourceFrames ||
        !std::all_of(output.waveform.begin(), output.waveform.end(),
                     [](float value) { return std::isfinite(value); })) {
      result.invalid_reason = "efficientphys_output_invalid";
      set_elapsed(&result, started);
      return result;
    }

    const TscanPostprocessResult post =
        postprocess_tscan_waveform(output.waveform);
    // The shared postprocessor remains the authority for validity and
    // confidence. Refine only a valid peak to remove the 6-second FFT-bin
    // quantization (70 BPM for an exact 72 BPM sinusoid) without changing the
    // established TSCAN numerical contract.
    result.bpm =
        post.is_valid ? refined_bpm(output.waveform, post.bpm) : post.bpm;
    result.confidence = post.confidence;
    result.is_valid = post.is_valid;
    result.invalid_reason = post.invalid_reason;
    result.waveform = std::move(output.waveform);
    set_elapsed(&result, started);
    return result;
  }

 private:
  std::unique_ptr<IEfficientPhysSession> session_;
  std::string backend_;
};

}  // namespace

std::unique_ptr<IDeepRuntime> make_onnxruntime_efficientphys_runtime(
    std::unique_ptr<IEfficientPhysSession> session) {
  if (!session) {
    throw std::invalid_argument("EfficientPhys session must not be null");
  }
  std::string backend = session->backend_name();
  if (backend != "onnxruntime_cpu") {
    throw std::invalid_argument(
        "EfficientPhys session backend must be onnxruntime_cpu");
  }
  return std::make_unique<EfficientPhysRuntime>(std::move(session),
                                                std::move(backend));
}

}  // namespace rppg_qnn
