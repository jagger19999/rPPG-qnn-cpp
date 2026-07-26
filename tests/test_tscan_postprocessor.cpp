#include "rppg_qnn/tscan_postprocessor.hpp"

#include "test_support.hpp"

#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kSamples = 180;
constexpr double kFps = 30.0;
constexpr double kPi = 3.14159265358979323846;
double max_confidence_error = 0.0;

std::vector<float> waveform(double dc, double frequency_hz,
                            double amplitude = 1.0,
                            std::size_t sample_count = kSamples) {
  std::vector<float> values(sample_count);
  for (std::size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<float>(
        dc + amplitude * std::sin(2.0 * kPi * frequency_hz *
                                  static_cast<double>(i) / kFps));
  }
  return values;
}

void expect_near(double actual, double expected, double tolerance) {
  EXPECT_TRUE(std::abs(actual - expected) < tolerance);
}

void expect_confidence_near(double actual, double expected) {
  max_confidence_error =
      std::max(max_confidence_error, std::abs(actual - expected));
  expect_near(actual, expected, 1e-4);
}

void matches_numpy_for_a_dc_offset_sinusoid() {
  const auto result = rppg_qnn::postprocess_tscan_waveform(waveform(3.0, 1.2));
  EXPECT_TRUE(result.is_valid);
  EXPECT_EQ(result.invalid_reason, std::string{});
  expect_near(result.bpm, 70.0, 1e-10);
  expect_confidence_near(result.confidence, 0.6299490739274951);
}

void matches_numpy_for_deterministic_mixed_waveforms() {
  std::vector<float> first(kSamples);
  std::vector<float> second(kSamples);
  for (std::size_t i = 0; i < kSamples; ++i) {
    const double t = static_cast<double>(i) / kFps;
    first[i] = static_cast<float>(
        11.0 + 1.3 * std::sin(2.0 * kPi * 1.05 * t) +
        0.7 * std::sin(2.0 * kPi * 1.8 * t) +
        0.2 * std::cos(2.0 * kPi * 0.4 * t));
    second[i] = static_cast<float>(
        -4.0 + 0.8 * std::cos(2.0 * kPi * 2.15 * t) +
        0.55 * std::sin(2.0 * kPi * 0.85 * t) +
        0.3 * std::sin(2.0 * kPi * 3.2 * t));
  }

  const auto first_result = rppg_qnn::postprocess_tscan_waveform(first);
  EXPECT_TRUE(first_result.is_valid);
  expect_near(first_result.bpm, 60.0, 1e-10);
  expect_confidence_near(first_result.confidence, 0.4574092009143706);

  const auto second_result = rppg_qnn::postprocess_tscan_waveform(second);
  EXPECT_TRUE(second_result.is_valid);
  expect_near(second_result.bpm, 130.0, 1e-10);
  expect_confidence_near(second_result.confidence, 0.46265211140762463);
}

void uses_inclusive_band_boundaries_and_ignores_out_of_band_power() {
  constexpr std::size_t boundary_samples = 240;
  const auto lower = rppg_qnn::postprocess_tscan_waveform(
      waveform(0.0, 0.75, 1.0, boundary_samples));
  EXPECT_TRUE(lower.is_valid);
  expect_near(lower.bpm, 45.0, 1e-10);
  expect_confidence_near(lower.confidence, 0.7979715513176878);

  const auto upper = rppg_qnn::postprocess_tscan_waveform(
      waveform(0.0, 2.5, 1.0, boundary_samples));
  EXPECT_TRUE(upper.is_valid);
  expect_near(upper.bpm, 150.0, 1e-10);
  expect_confidence_near(upper.confidence, 0.797994505318668);

  std::vector<float> dominated(kSamples);
  for (std::size_t i = 0; i < kSamples; ++i) {
    const double t = static_cast<double>(i) / kFps;
    dominated[i] = static_cast<float>(
        10.0 * std::sin(2.0 * kPi * 0.5 * t) +
        std::sin(2.0 * kPi * 1.5 * t));
  }
  const auto result = rppg_qnn::postprocess_tscan_waveform(dominated);
  EXPECT_TRUE(result.is_valid);
  expect_near(result.bpm, 90.0, 1e-10);
  expect_confidence_near(result.confidence, 0.6645100102850933);
}

void rejects_structurally_invalid_waveforms() {
  for (const auto& values :
       {std::vector<float>{}, std::vector<float>(7U, 1.0F)}) {
    const auto result = rppg_qnn::postprocess_tscan_waveform(values);
    EXPECT_TRUE(!result.is_valid);
    EXPECT_EQ(result.invalid_reason, std::string{"TSCAN_WAVEFORM_INVALID"});
    EXPECT_EQ(result.bpm, 0.0);
    EXPECT_EQ(result.confidence, 0.0);
  }

  auto nonfinite = waveform(0.0, 1.0);
  nonfinite[42] = std::numeric_limits<float>::infinity();
  const auto result = rppg_qnn::postprocess_tscan_waveform(nonfinite);
  EXPECT_TRUE(!result.is_valid);
  EXPECT_EQ(result.invalid_reason, std::string{"TSCAN_WAVEFORM_INVALID"});
}

void preserves_python_constant_and_threshold_semantics() {
  const auto constant =
      rppg_qnn::postprocess_tscan_waveform(std::vector<float>(kSamples, 7.0F));
  EXPECT_TRUE(constant.is_valid);
  expect_near(constant.bpm, 50.0, 1e-10);
  EXPECT_EQ(constant.confidence, 0.0);

  const auto high_threshold =
      rppg_qnn::postprocess_tscan_waveform(waveform(3.0, 1.2), 1.0);
  EXPECT_TRUE(high_threshold.is_valid);
  expect_near(high_threshold.bpm, 70.0, 1e-10);
}

void rejects_invalid_thresholds() {
  for (double threshold :
       {-0.01, 1.01, std::numeric_limits<double>::infinity(),
        std::numeric_limits<double>::quiet_NaN()}) {
    try {
      (void)rppg_qnn::postprocess_tscan_waveform(waveform(0.0, 1.0), threshold);
      EXPECT_TRUE(false);
    } catch (const std::invalid_argument&) {
    } catch (...) {
      EXPECT_TRUE(false);
    }
  }
}

}  // namespace

int main() {
  matches_numpy_for_a_dc_offset_sinusoid();
  matches_numpy_for_deterministic_mixed_waveforms();
  uses_inclusive_band_boundaries_and_ignores_out_of_band_power();
  rejects_structurally_invalid_waveforms();
  preserves_python_constant_and_threshold_semantics();
  rejects_invalid_thresholds();
  std::cout << "TSCAN postprocessing confidence max_abs="
            << max_confidence_error << '\n';
  return test_support::finish();
}
