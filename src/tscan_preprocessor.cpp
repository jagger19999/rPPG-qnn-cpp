#include "rppg_qnn/tscan_preprocessor.hpp"

#include "rppg_qnn/error.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace rppg_qnn {
namespace {

constexpr std::size_t kFrames = 180;
constexpr std::size_t kHeight = 72;
constexpr std::size_t kWidth = 72;
constexpr std::size_t kRgbChannels = 3;
constexpr std::size_t kOutputChannels = 6;
constexpr std::size_t kPixels = kHeight * kWidth;
constexpr std::size_t kInputSize = kFrames * kPixels * kRgbChannels;
constexpr std::size_t kDiffSize = (kFrames - 1U) * kPixels * kRgbChannels;

[[noreturn]] void fail(const char* message) {
  throw AppError(ErrorCode::InferenceFailed, message);
}

std::size_t input_offset(std::size_t frame, std::size_t pixel,
                         std::size_t channel) {
  return (frame * kPixels + pixel) * kRgbChannels + channel;
}

std::size_t output_offset(std::size_t frame, std::size_t channel,
                          std::size_t pixel) {
  return (frame * kOutputChannels + channel) * kPixels + pixel;
}

double population_std(const std::vector<float>& values) {
  double sum = 0.0;
  for (float value : values) {
    sum += static_cast<double>(value);
  }
  const double mean = sum / static_cast<double>(values.size());
  double squared_deviations = 0.0;
  for (float value : values) {
    const double deviation = static_cast<double>(value) - mean;
    squared_deviations += deviation * deviation;
  }
  return std::sqrt(squared_deviations / static_cast<double>(values.size()));
}

}  // namespace

TscanPreprocessor::TscanPreprocessor()
    : differences_(kDiffSize),
      output_{std::vector<float>(kFrames * kOutputChannels * kPixels, 0.0F),
              {180, 6, 72, 72}} {}

const TscanTensor& TscanPreprocessor::preprocess(const DeepInput& input) {
  if (input.shape != std::vector<std::int64_t>{180, 72, 72, 3}) {
    fail("TSCAN_PREPROCESS_SHAPE expected [180,72,72,3]");
  }
  if (input.tensor.size() != kInputSize) {
    fail("TSCAN_PREPROCESS_LENGTH does not match shape");
  }
  for (float value : input.tensor) {
    if (!std::isfinite(value)) {
      fail("TSCAN_PREPROCESS_NONFINITE input contains nonfinite value");
    }
  }

  std::size_t difference_index = 0;
  for (std::size_t frame = 0; frame + 1U < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        const float previous = input.tensor[input_offset(frame, pixel, channel)];
        const float next = input.tensor[input_offset(frame + 1U, pixel, channel)];
        differences_[difference_index++] =
            (next - previous) / (next + previous + 1e-7F);
      }
    }
  }

  const double difference_scale = population_std(differences_);
  const double appearance_scale = population_std(input.tensor);
  if (!std::isfinite(difference_scale) || difference_scale == 0.0) {
    fail("TSCAN_PREPROCESS_VARIANCE_DIFF diff scale is zero or nonfinite");
  }
  if (!std::isfinite(appearance_scale) || appearance_scale == 0.0) {
    fail("TSCAN_PREPROCESS_VARIANCE_APPEARANCE appearance scale is zero or nonfinite");
  }
  double appearance_sum = 0.0;
  for (float value : input.tensor) {
    appearance_sum += static_cast<double>(value);
  }
  const double appearance_mean =
      appearance_sum / static_cast<double>(input.tensor.size());

  difference_index = 0;
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        if (frame + 1U < kFrames) {
          output_.values[output_offset(frame, channel, pixel)] =
              static_cast<float>(static_cast<double>(differences_[difference_index++]) /
                                 difference_scale);
        }
        output_.values[output_offset(frame, channel + 3U, pixel)] =
            static_cast<float>((static_cast<double>(
                                    input.tensor[input_offset(frame, pixel, channel)]) -
                                appearance_mean) /
                               appearance_scale);
      }
    }
  }
  return output_;
}

TscanTensor preprocess_tscan_rgb(const DeepInput& input) {
  TscanPreprocessor preprocessor;
  return preprocessor.preprocess(input);
}

}  // namespace rppg_qnn
