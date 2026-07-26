#include "rppg_qnn/tscan_preprocessor.hpp"

#include "test_support.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kFrames = 180;
constexpr std::size_t kHeight = 72;
constexpr std::size_t kWidth = 72;
constexpr std::size_t kRgbChannels = 3;
constexpr std::size_t kPixels = kHeight * kWidth;
constexpr std::size_t kInputSize = kFrames * kPixels * kRgbChannels;

rppg_qnn::DeepInput input_with(float value) {
  rppg_qnn::DeepInput input;
  input.shape = {180, 72, 72, 3};
  input.tensor.assign(kInputSize, value);
  return input;
}

std::size_t input_offset(std::size_t frame, std::size_t pixel,
                         std::size_t channel) {
  return (frame * kPixels + pixel) * kRgbChannels + channel;
}

std::size_t output_offset(std::size_t frame, std::size_t channel,
                          std::size_t pixel) {
  return (frame * 6U + channel) * kPixels + pixel;
}

void expect_near(float actual, float expected, float tolerance) {
  EXPECT_TRUE(std::abs(actual - expected) <= tolerance);
}

bool finite_absolute_error(float actual, float expected, double& error) {
  if (!std::isfinite(actual) || !std::isfinite(expected)) {
    error = std::numeric_limits<double>::quiet_NaN();
    return false;
  }
  error = std::abs(static_cast<double>(actual) -
                   static_cast<double>(expected));
  return std::isfinite(error);
}

void rejects_a_nonfinite_full_tensor_comparison() {
  double error = 0.0;
  EXPECT_TRUE(!finite_absolute_error(
      std::numeric_limits<float>::quiet_NaN(), 0.0F, error));
  EXPECT_TRUE(!finite_absolute_error(
      0.0F, std::numeric_limits<float>::infinity(), error));
}

void expect_preprocess_error(const rppg_qnn::DeepInput& input,
                             const std::string& prefix) {
  try {
    (void)rppg_qnn::preprocess_tscan_rgb(input);
    EXPECT_TRUE(false);
  } catch (const rppg_qnn::AppError& error) {
    EXPECT_EQ(error.code(), rppg_qnn::ErrorCode::InferenceFailed);
    EXPECT_TRUE(std::string(error.what()).rfind(prefix, 0) == 0);
  }
}

void expect_preprocess_success(const rppg_qnn::DeepInput& input) {
  try {
    const auto output = rppg_qnn::preprocess_tscan_rgb(input);
    for (float value : output.values) {
      EXPECT_TRUE(std::isfinite(value));
    }
  } catch (const rppg_qnn::AppError&) {
    EXPECT_TRUE(false);
  }
}

double reference_population_std(const std::vector<float>& values);

void validates_shape_length_finiteness_and_variance() {
  auto wrong_shape = input_with(1.0F);
  wrong_shape.shape = {180, 72, 72, 1};
  expect_preprocess_error(wrong_shape, "TSCAN_PREPROCESS_SHAPE");

  auto wrong_length = input_with(1.0F);
  wrong_length.tensor.pop_back();
  expect_preprocess_error(wrong_length, "TSCAN_PREPROCESS_LENGTH");

  auto nonfinite = input_with(1.0F);
  nonfinite.tensor[13] = std::numeric_limits<float>::infinity();
  expect_preprocess_error(nonfinite, "TSCAN_PREPROCESS_NONFINITE");

  expect_preprocess_error(input_with(7.0F), "TSCAN_PREPROCESS_VARIANCE");
}

void accepts_positive_scales_below_the_previous_guard() {
  auto low_appearance_scale = input_with(0.0F);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        low_appearance_scale.tensor[input_offset(frame, pixel, channel)] =
            1e-10F * static_cast<float>(frame + pixel % 5U + channel * 10U);
      }
    }
  }
  const double appearance_scale =
      reference_population_std(low_appearance_scale.tensor);
  EXPECT_TRUE(appearance_scale > 0.0);
  EXPECT_TRUE(appearance_scale < 1e-7);
  expect_preprocess_success(low_appearance_scale);

  auto low_diff_scale = input_with(0.0F);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        low_diff_scale.tensor[input_offset(frame, pixel, channel)] =
            1.0F + static_cast<float>(pixel % 5U + channel) +
            1e-7F * static_cast<float>(frame);
      }
    }
  }
  std::vector<float> differences;
  differences.reserve((kFrames - 1U) * kPixels * kRgbChannels);
  for (std::size_t frame = 0; frame + 1U < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        const float previous =
            low_diff_scale.tensor[input_offset(frame, pixel, channel)];
        const float next =
            low_diff_scale.tensor[input_offset(frame + 1U, pixel, channel)];
        differences.push_back((next - previous) /
                              (next + previous + 1e-7F));
      }
    }
  }
  const double difference_scale = reference_population_std(differences);
  EXPECT_TRUE(difference_scale > 0.0);
  EXPECT_TRUE(difference_scale < 1e-7);
  expect_preprocess_success(low_diff_scale);
}

double reference_population_std(const std::vector<float>& values) {
  const double mean =
      std::accumulate(values.begin(), values.end(), 0.0) /
      static_cast<double>(values.size());
  double squared_deviations = 0.0;
  for (float value : values) {
    const double deviation = static_cast<double>(value) - mean;
    squared_deviations += deviation * deviation;
  }
  return std::sqrt(squared_deviations / static_cast<double>(values.size()));
}

void matches_a_manually_calculable_layout_and_rgb_order() {
  auto input = input_with(0.0F);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        input.tensor[input_offset(frame, pixel, channel)] =
            static_cast<float>(10U * channel + frame + pixel % 5U);
      }
    }
  }
  const auto output = rppg_qnn::preprocess_tscan_rgb(input);
  EXPECT_EQ(output.shape, (std::vector<std::int64_t>{180, 6, 72, 72}));
  EXPECT_EQ(output.values.size(), kFrames * 6U * kPixels);

  std::vector<float> reference_differences;
  reference_differences.reserve((kFrames - 1U) * kPixels * kRgbChannels);
  for (std::size_t frame = 0; frame + 1U < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        const float previous = input.tensor[input_offset(frame, pixel, channel)];
        const float next = input.tensor[input_offset(frame + 1U, pixel, channel)];
        reference_differences.push_back(
            (next - previous) / (next + previous + 1e-7F));
      }
    }
  }
  const double difference_scale = reference_population_std(reference_differences);
  const double appearance_mean =
      std::accumulate(input.tensor.begin(), input.tensor.end(), 0.0) /
      static_cast<double>(input.tensor.size());
  const double appearance_scale = reference_population_std(input.tensor);
  constexpr std::size_t checked_frame = 37;
  constexpr std::size_t checked_pixel = 73;
  for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
    const float previous =
        input.tensor[input_offset(checked_frame, checked_pixel, channel)];
    const float next =
        input.tensor[input_offset(checked_frame + 1U, checked_pixel, channel)];
    const float raw_difference =
        (next - previous) / (next + previous + 1e-7F);
    expect_near(output.values[output_offset(checked_frame, channel,
                                            checked_pixel)],
                static_cast<float>(raw_difference / difference_scale), 1e-5F);
    expect_near(output.values[output_offset(checked_frame, channel + 3U,
                                            checked_pixel)],
                static_cast<float>((static_cast<double>(previous) -
                                    appearance_mean) /
                                   appearance_scale),
                1e-5F);
  }

  for (std::size_t channel = 0; channel < 3; ++channel) {
    const float early = output.values[output_offset(0, channel, 0)];
    const float later = output.values[output_offset(0, channel, 4)];
    EXPECT_TRUE(early > later);
    const float appearance = output.values[output_offset(0, channel + 3U, 0)];
    if (channel != 0) {
      EXPECT_TRUE(appearance >
                  output.values[output_offset(0, channel + 2U, 0)]);
    }
  }
  for (std::size_t channel = 0; channel < 3; ++channel) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      EXPECT_EQ(output.values[output_offset(179, channel, pixel)], 0.0F);
    }
  }
  for (float value : output.values) {
    EXPECT_TRUE(std::isfinite(value));
  }

  const auto repeated = rppg_qnn::preprocess_tscan_rgb(input);
  EXPECT_EQ(output.values, repeated.values);
}

void distinguishes_zero_diff_from_the_unreachable_appearance_only_case() {
  auto identical_frames = input_with(0.0F);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        identical_frames.tensor[input_offset(frame, pixel, channel)] =
            static_cast<float>(pixel % 5U + channel * 10U);
      }
    }
  }
  expect_preprocess_error(identical_frames, "TSCAN_PREPROCESS_VARIANCE_DIFF");
  expect_preprocess_error(input_with(7.0F), "TSCAN_PREPROCESS_VARIANCE_DIFF");
}

struct Checkpoint {
  std::size_t frame;
  std::size_t channel;
  std::array<float, 4> expected;
};

void matches_python_reference_checkpoints() {
  auto input = input_with(0.0F);
  constexpr float kAngularNumerator =
      static_cast<float>(2.0 * 3.14159265358979323846 * 1.2);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    const float phase =
        (kAngularNumerator * static_cast<float>(frame)) / 30.0F;
    const float pulse = 2.5F * std::sin(phase);
    for (std::size_t y = 0; y < kHeight; ++y) {
      for (std::size_t x = 0; x < kWidth; ++x) {
        const std::size_t pixel = y * kWidth + x;
        for (std::size_t channel = 0; channel < 3; ++channel) {
          input.tensor[input_offset(frame, pixel, channel)] =
              80.0F + 0.31F * static_cast<float>(x) +
              0.17F * static_cast<float>(y) + 11.0F * static_cast<float>(channel) +
              pulse * static_cast<float>(channel + 1U);
        }
      }
    }
  }

  const auto output = rppg_qnn::preprocess_tscan_rgb(input);
  constexpr std::array<std::size_t, 4> pixels{0, 73, 2591, 5183};
  const std::array<Checkpoint, 30> checkpoints{{
      {0,0,{0.908403993F,0.903006911F,0.673816681F,0.637766898F}}, {0,1,{1.59251022F,1.58421075F,1.22015738F,1.16075206F}}, {0,2,{2.12627745F,2.11640811F,1.67208278F,1.59739518F}},
      {0,3,{-2.29818583F,-2.25893283F,-0.0116753401F,0.488805741F}}, {0,4,{-1.39862871F,-1.35937572F,0.887881756F,1.38836288F}}, {0,5,{-0.499071658F,-0.459818602F,1.78743827F,2.28792F}},
      {1,0,{0.844982803F,0.839999735F,0.627980411F,0.594558895F}}, {1,1,{1.47308969F,1.46551156F,1.13209367F,1.07749879F}}, {1,2,{1.95830584F,1.94937229F,1.5456934F,1.47754967F}},
      {1,3,{-2.24734235F,-2.20808935F,0.0391681977F,0.539649248F}}, {1,4,{-1.29694223F,-1.25768924F,0.989568174F,1.49004924F}}, {1,5,{-0.346541673F,-0.307288617F,1.93996823F,2.44044995F}},
      {89,0,{-0.817702115F,-0.812751651F,-0.603559613F,-0.570837915F}}, {89,1,{-1.45443952F,-1.44660342F,-1.10558438F,-1.0504359F}}, {89,2,{-1.96430087F,-1.95476699F,-1.52970779F,-1.45905685F}},
      {89,3,{-2.37344694F,-2.33419394F,-0.0869364515F,0.413544625F}}, {89,4,{-1.54915094F,-1.50989795F,0.737359524F,1.23784065F}}, {89,5,{-0.724855006F,-0.68560195F,1.56165493F,2.06213713F}},
      {178,0,{0.572247624F,0.568913817F,0.426622033F,0.404112339F}}, {178,1,{0.988659561F,0.983680069F,0.763521671F,0.727271259F}}, {178,2,{1.30525482F,1.29946518F,1.03631008F,0.99158746F}},
      {178,3,{-2.15823388F,-2.11898088F,0.128276512F,0.628757596F}}, {178,4,{-1.11872506F,-1.07947195F,1.16778541F,1.66826653F}}, {178,5,{-0.0792154968F,-0.0399624445F,2.20729375F,2.70777535F}},
      {179,0,{0,0,0,0}}, {179,1,{0,0,0,0}}, {179,2,{0,0,0,0}},
      {179,3,{-2.12556696F,-2.08631396F,0.160943493F,0.661424577F}}, {179,4,{-1.0533911F,-1.01413798F,1.23311937F,1.7336005F}}, {179,5,{0.018784862F,0.0580379143F,2.30529475F,2.80577636F}},
  }};
  for (const auto& checkpoint : checkpoints) {
    for (std::size_t index = 0; index < pixels.size(); ++index) {
      expect_near(output.values[output_offset(checkpoint.frame, checkpoint.channel,
                                              pixels[index])],
                  checkpoint.expected[index], 1e-5F);
    }
  }
}

void matches_the_full_analytic_python_tensor() {
  auto input = input_with(0.0F);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    const float t = static_cast<float>(frame);
    const float pulse = 2.5F * std::sin(
        static_cast<float>(2.0 * 3.14159265358979323846 * 1.2) * t / 30.0F);
    for (std::size_t y = 0; y < kHeight; ++y) {
      for (std::size_t x = 0; x < kWidth; ++x) {
        const std::size_t pixel = y * kWidth + x;
        for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
          input.tensor[input_offset(frame, pixel, channel)] =
              80.0F + 0.31F * static_cast<float>(x) +
              0.17F * static_cast<float>(y) +
              11.0F * static_cast<float>(channel) +
              pulse * static_cast<float>(channel + 1U);
        }
      }
    }
  }

  std::vector<float> raw_diff((kFrames - 1U) * kPixels * kRgbChannels);
  double appearance_sum = 0.0;
  double appearance_square_sum = 0.0;
  double diff_sum = 0.0;
  double diff_square_sum = 0.0;
  std::size_t diff_index = 0;
  for (std::size_t index = 0; index < input.tensor.size(); ++index) {
    const double value = input.tensor[index];
    appearance_sum += value;
    appearance_square_sum += value * value;
  }
  for (std::size_t frame = 0; frame + 1U < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        const float before = input.tensor[input_offset(frame, pixel, channel)];
        const float after = input.tensor[input_offset(frame + 1U, pixel, channel)];
        const float value = (after - before) / (after + before + 1e-7F);
        raw_diff[diff_index++] = value;
        diff_sum += value;
        diff_square_sum += static_cast<double>(value) * value;
      }
    }
  }
  const double appearance_count = static_cast<double>(input.tensor.size());
  const double appearance_mean = appearance_sum / appearance_count;
  const double appearance_scale = std::sqrt(
      appearance_square_sum / appearance_count - appearance_mean * appearance_mean);
  const double diff_count = static_cast<double>(raw_diff.size());
  const double diff_mean = diff_sum / diff_count;
  const double diff_scale =
      std::sqrt(diff_square_sum / diff_count - diff_mean * diff_mean);

  const auto actual = rppg_qnn::preprocess_tscan_rgb(input);
  double max_abs = 0.0;
  diff_index = 0;
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    for (std::size_t channel = 0; channel < 6U; ++channel) {
      for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
        float expected = 0.0F;
        if (channel < kRgbChannels) {
          if (frame + 1U < kFrames) {
            const std::size_t index =
                (frame * kPixels + pixel) * kRgbChannels + channel;
            expected = static_cast<float>(raw_diff[index] / diff_scale);
          }
        } else {
          const float source = input.tensor[
              (frame * kPixels + pixel) * kRgbChannels +
              (channel - kRgbChannels)];
          expected = static_cast<float>((source - appearance_mean) /
                                        appearance_scale);
        }
        const float actual_value =
            actual.values[output_offset(frame, channel, pixel)];
        EXPECT_TRUE(std::isfinite(actual_value));
        EXPECT_TRUE(std::isfinite(expected));
        double error = std::numeric_limits<double>::quiet_NaN();
        EXPECT_TRUE(finite_absolute_error(actual_value, expected, error));
        EXPECT_TRUE(std::isfinite(error));
        max_abs = std::max(max_abs, error);
      }
    }
  }
  std::cout << "TSCAN full preprocessing max_abs=" << max_abs << '\n';
  EXPECT_TRUE(max_abs < 1e-5);
}

void worker_local_preprocessor_has_no_cross_call_contamination() {
  auto first_input = input_with(0.0F);
  auto second_input = input_with(0.0F);
  for (std::size_t frame = 0; frame < kFrames; ++frame) {
    for (std::size_t pixel = 0; pixel < kPixels; ++pixel) {
      for (std::size_t channel = 0; channel < kRgbChannels; ++channel) {
        const std::size_t offset = input_offset(frame, pixel, channel);
        first_input.tensor[offset] =
            40.0F + static_cast<float>(frame) * 0.17F +
            static_cast<float>(pixel % 13U) * 0.11F +
            static_cast<float>(channel) * 3.0F;
        second_input.tensor[offset] =
            90.0F + static_cast<float>((frame * frame + pixel) % 97U) * 0.07F +
            static_cast<float>(channel) * 5.0F;
      }
    }
  }

  rppg_qnn::TscanPreprocessor preprocessor;
  const auto first = preprocessor.preprocess(first_input);
  const std::vector<float> first_values = first.values;
  const auto second = preprocessor.preprocess(second_input);
  const std::vector<float> second_values = second.values;
  const auto repeated = preprocessor.preprocess(first_input);

  EXPECT_EQ(first.shape, (std::vector<std::int64_t>{180, 6, 72, 72}));
  EXPECT_EQ(second.shape, first.shape);
  EXPECT_TRUE(first_values != second_values);
  EXPECT_EQ(repeated.values, first_values);
  EXPECT_EQ(rppg_qnn::preprocess_tscan_rgb(first_input).values, first_values);
}

}  // namespace

int main() {
  rejects_a_nonfinite_full_tensor_comparison();
  validates_shape_length_finiteness_and_variance();
  accepts_positive_scales_below_the_previous_guard();
  matches_a_manually_calculable_layout_and_rgb_order();
  distinguishes_zero_diff_from_the_unreachable_appearance_only_case();
  matches_python_reference_checkpoints();
  matches_the_full_analytic_python_tensor();
  worker_local_preprocessor_has_no_cross_call_contamination();
  return test_support::finish();
}
