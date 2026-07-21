#include "rppg_qnn/deep_runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <string>
#include <thread>
#include <utility>

namespace rppg_qnn {
namespace {

constexpr double kSampleRateHz = 30.0;
constexpr double kMinimumHz = 0.7;
constexpr double kMaximumHz = 3.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kConfidenceThreshold = 0.10;
constexpr double kPowerEpsilon = 1e-12;
constexpr std::int64_t kMaximumDimension = 4096;
constexpr std::int64_t kMaximumFrames = 4096;

HeartRateResult invalid_result(const DeepInput& input, const char* reason) {
  HeartRateResult result;
  result.method = "FAKE_DEEP";
  result.backend = "fake";
  result.window_start_sec = std::isfinite(input.start_sec) ? input.start_sec : 0.0;
  result.window_end_sec = std::isfinite(input.end_sec) ? input.end_sec : 0.0;
  result.source_fps = std::isfinite(input.source_fps) ? input.source_fps : 0.0;
  result.source_frame_count = input.source_frame_count;
  result.max_frame_gap_sec =
      std::isfinite(input.max_frame_gap_sec) ? input.max_frame_gap_sec : 0.0;
  result.invalid_reason = reason;
  return result;
}

bool valid_shape(const DeepInput& input, std::size_t* frame_pixels) {
  if (input.shape.size() != 4U || input.shape[0] <= 0 || input.shape[1] <= 0 ||
      input.shape[2] <= 0 || input.shape[3] != 3 ||
      input.shape[0] > kMaximumFrames || input.shape[1] > kMaximumDimension ||
      input.shape[2] > kMaximumDimension) {
    return false;
  }
  const std::uint64_t frames = static_cast<std::uint64_t>(input.shape[0]);
  const std::uint64_t height = static_cast<std::uint64_t>(input.shape[1]);
  const std::uint64_t width = static_cast<std::uint64_t>(input.shape[2]);
  const std::uint64_t pixels = frames * height * width * 3U;
  if (pixels > std::numeric_limits<std::size_t>::max()) {
    return false;
  }
  *frame_pixels = static_cast<std::size_t>(height * width * 3U);
  return input.tensor.size() == static_cast<std::size_t>(pixels);
}

class FakeDeepRuntime final : public IDeepRuntime {
 public:
  explicit FakeDeepRuntime(std::chrono::milliseconds latency) : latency_(latency) {}

  [[nodiscard]] std::string backend_name() const override { return "fake"; }

  HeartRateResult infer(const DeepInput& input) override {
    const auto started = std::chrono::steady_clock::now();
    if (latency_.count() > 0) {
      std::this_thread::sleep_for(latency_);
    }
    std::size_t frame_pixels = 0;
    if (!std::isfinite(input.start_sec) || !std::isfinite(input.end_sec) ||
        input.end_sec <= input.start_sec || !std::isfinite(input.source_fps) ||
        input.source_fps < 0.0 || !std::isfinite(input.max_frame_gap_sec) ||
        input.max_frame_gap_sec < 0.0 || !valid_shape(input, &frame_pixels) ||
        !std::all_of(input.tensor.begin(), input.tensor.end(),
                     [](float value) { return std::isfinite(value); })) {
      HeartRateResult result = invalid_result(input, "model_input_invalid");
      result.inference_ms = std::chrono::duration<double, std::milli>(
                              std::chrono::steady_clock::now() - started)
                              .count();
      return result;
    }

    const std::size_t frames = static_cast<std::size_t>(input.shape[0]);
    std::vector<float> waveform;
    waveform.reserve(frames);
    for (std::size_t frame = 0; frame < frames; ++frame) {
      double green_sum = 0.0;
      const std::size_t offset = frame * frame_pixels;
      for (std::size_t pixel = 0; pixel < frame_pixels; pixel += 3U) {
        green_sum += input.tensor[offset + pixel + 1U];
      }
      waveform.push_back(static_cast<float>(green_sum /
                                            static_cast<double>(frame_pixels / 3U)));
    }
    const double mean = std::accumulate(waveform.begin(), waveform.end(), 0.0) /
                        static_cast<double>(waveform.size());
    double scale = 0.0;
    for (float& value : waveform) {
      value = static_cast<float>(static_cast<double>(value) - mean);
      scale = std::max(scale, std::abs(static_cast<double>(value)));
    }
    for (float& value : waveform) {
      value = scale > 0.0 ? static_cast<float>(value / scale) : 0.0F;
    }

    HeartRateResult result = invalid_result(input, "low_confidence");
    result.waveform = waveform;
    const int first_bin = std::max(1, static_cast<int>(std::ceil(
        kMinimumHz * static_cast<double>(frames) / kSampleRateHz)));
    const int last_bin = std::min(static_cast<int>(frames / 2U),
                                 static_cast<int>(std::floor(
        kMaximumHz * static_cast<double>(frames) / kSampleRateHz)));
    double total_power = 0.0;
    double peak_power = 0.0;
    int peak_bin = first_bin;
    for (int bin = 1; bin <= static_cast<int>(frames / 2U); ++bin) {
      double real = 0.0;
      double imaginary = 0.0;
      for (std::size_t index = 0; index < frames; ++index) {
        const double phase = 2.0 * kPi * static_cast<double>(bin * index) /
                             static_cast<double>(frames);
        real += waveform[index] * std::cos(phase);
        imaginary -= waveform[index] * std::sin(phase);
      }
      const double power = real * real + imaginary * imaginary;
      if (!std::isfinite(power)) {
        result.inference_ms = std::chrono::duration<double, std::milli>(
                                std::chrono::steady_clock::now() - started)
                                .count();
        return result;
      }
      total_power += power;
      if (bin >= first_bin && bin <= last_bin && power > peak_power) {
        peak_power = power;
        peak_bin = bin;
      }
    }
    if (std::isfinite(peak_power) && std::isfinite(total_power) &&
        peak_power > kPowerEpsilon && total_power > kPowerEpsilon) {
      const double confidence = peak_power / total_power;
      if (std::isfinite(confidence)) {
        result.confidence = confidence;
        if (confidence >= kConfidenceThreshold) {
          result.is_valid = true;
          result.invalid_reason.clear();
          result.bpm = static_cast<double>(peak_bin) * kSampleRateHz * 60.0 /
                       static_cast<double>(frames);
        }
      }
    }
    result.inference_ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - started)
                            .count();
    return result;
  }

 private:
  std::chrono::milliseconds latency_;
};

}  // namespace

std::unique_ptr<IDeepRuntime> make_fake_deep_runtime(
    std::chrono::milliseconds latency) {
  return std::make_unique<FakeDeepRuntime>(latency);
}

}  // namespace rppg_qnn
