#include "rppg_qnn/deep_stabilizer.hpp"

#include "test_support.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::size_t kSamples = 180U;
constexpr double kFps = 30.0;
constexpr double kPi = 3.14159265358979323846;

std::vector<float> waveform(
    std::initializer_list<std::pair<double, double>> bpm_amplitudes) {
  std::vector<float> values(kSamples, 0.0F);
  for (std::size_t sample = 0; sample < values.size(); ++sample) {
    const double time = static_cast<double>(sample) / kFps;
    for (const auto& [bpm, amplitude] : bpm_amplitudes) {
      values[sample] += static_cast<float>(
          amplitude * std::sin(2.0 * kPi * (bpm / 60.0) * time));
    }
  }
  return values;
}

void expect_near(double actual, double expected, double tolerance = 1e-9) {
  EXPECT_TRUE(std::abs(actual - expected) < tolerance);
}

void rejects_low_confidence() {
  rppg_qnn::DeepStabilizer stabilizer;
  const auto result = stabilizer.stabilize(waveform({{70.0, 1.0}}), 70.0,
                                                    0.099);
  EXPECT_TRUE(!result.stability_valid);
  EXPECT_EQ(result.display_bpm, 0.0);
  EXPECT_EQ(result.correction_reason,
            std::string("rejected_low_confidence"));
}

void rejects_nonfinite_and_constant_waveforms() {
  rppg_qnn::DeepStabilizer stabilizer;
  auto nonfinite = waveform({{70.0, 1.0}});
  nonfinite[4] = std::numeric_limits<float>::quiet_NaN();
  auto result = stabilizer.stabilize(nonfinite, 70.0, 0.8);
  EXPECT_TRUE(!result.stability_valid);
  EXPECT_EQ(result.display_bpm, 0.0);
  EXPECT_EQ(result.correction_reason,
            std::string("rejected_nonfinite_waveform"));

  result = stabilizer.stabilize(std::vector<float>(kSamples, 4.0F), 70.0,
                                0.8);
  EXPECT_TRUE(!result.stability_valid);
  EXPECT_EQ(result.display_bpm, 0.0);
  EXPECT_EQ(result.correction_reason,
            std::string("rejected_constant_waveform"));
}

void rejects_nonfinite_raw_bpm() {
  rppg_qnn::DeepStabilizer stabilizer;
  const auto result = stabilizer.stabilize(
      waveform({{70.0, 1.0}}),
      std::numeric_limits<double>::quiet_NaN(), 0.8);
  EXPECT_TRUE(!result.stability_valid);
  EXPECT_EQ(result.display_bpm, 0.0);
  EXPECT_EQ(result.correction_reason, std::string("rejected_invalid_bpm"));
}

void corrects_half_and_double_only_with_spectral_support() {
  rppg_qnn::DeepStabilizer half_stabilizer;
  (void)half_stabilizer.stabilize(waveform({{70.0, 1.0}}), 70.0, 0.8);
  auto result = half_stabilizer.stabilize(
      waveform({{140.0, 1.0}, {70.0, 0.8}}), 140.0, 0.8);
  EXPECT_TRUE(result.stability_valid);
  expect_near(result.display_bpm, 70.0);
  EXPECT_EQ(result.correction_reason, std::string("corrected_to_half"));

  rppg_qnn::DeepStabilizer double_stabilizer;
  (void)double_stabilizer.stabilize(waveform({{120.0, 1.0}}), 120.0, 0.8);
  result = double_stabilizer.stabilize(
      waveform({{60.0, 1.0}, {120.0, 0.8}}), 60.0, 0.8);
  EXPECT_TRUE(result.stability_valid);
  expect_near(result.display_bpm, 120.0);
  EXPECT_EQ(result.correction_reason, std::string("corrected_to_double"));

  rppg_qnn::DeepStabilizer unsupported;
  (void)unsupported.stabilize(waveform({{70.0, 1.0}}), 70.0, 0.8);
  result = unsupported.stabilize(
      waveform({{140.0, 1.0}, {70.0, 0.3}}), 140.0, 0.8);
  EXPECT_TRUE(result.stability_valid);
  expect_near(result.display_bpm, 140.0);
  EXPECT_EQ(result.correction_reason,
            std::string("accepted_supported_jump"));
}

void rejects_unsupported_large_jump() {
  rppg_qnn::DeepStabilizer stabilizer;
  (void)stabilizer.stabilize(waveform({{70.0, 1.0}}), 70.0, 0.8);
  const auto result = stabilizer.stabilize(
      waveform({{110.0, 1.0}, {70.0, 0.9}}), 110.0, 0.8);
  EXPECT_TRUE(!result.stability_valid);
  EXPECT_EQ(result.display_bpm, 0.0);
  EXPECT_EQ(result.correction_reason,
            std::string("rejected_unsupported_jump"));
}

void allows_supported_large_jump_and_clears_history() {
  rppg_qnn::DeepStabilizer stabilizer;
  (void)stabilizer.stabilize(waveform({{70.0, 1.0}}), 70.0, 0.8);
  auto result = stabilizer.stabilize(
      waveform({{110.0, 1.0}, {70.0, 0.4}}), 110.0, 0.8);
  EXPECT_TRUE(result.stability_valid);
  expect_near(result.display_bpm, 110.0);
  EXPECT_EQ(result.correction_reason,
            std::string("accepted_supported_jump"));

  result = stabilizer.stabilize(waveform({{108.0, 1.0}}), 108.0, 0.8);
  EXPECT_TRUE(result.stability_valid);
  expect_near(result.display_bpm, 109.0);
}

void rejected_value_does_not_pollute_history() {
  rppg_qnn::DeepStabilizer stabilizer;
  (void)stabilizer.stabilize(waveform({{70.0, 1.0}}), 70.0, 0.8);
  const auto rejected = stabilizer.stabilize(
      waveform({{110.0, 1.0}, {70.0, 0.9}}), 110.0, 0.8);
  EXPECT_TRUE(!rejected.stability_valid);

  const auto accepted =
      stabilizer.stabilize(waveform({{72.0, 1.0}}), 72.0, 0.8);
  EXPECT_TRUE(accepted.stability_valid);
  expect_near(accepted.display_bpm, 71.0);
}

void displays_median_of_three_accepted_values() {
  rppg_qnn::DeepStabilizer stabilizer;
  (void)stabilizer.stabilize(waveform({{70.0, 1.0}}), 70.0, 0.8);
  (void)stabilizer.stabilize(waveform({{90.0, 1.0}}), 90.0, 0.8);
  const auto result =
      stabilizer.stabilize(waveform({{80.0, 1.0}}), 80.0, 0.8);
  EXPECT_TRUE(result.stability_valid);
  expect_near(result.display_bpm, 80.0);
  EXPECT_EQ(result.correction_reason, std::string("accepted_raw"));
}

void reset_discards_history() {
  rppg_qnn::DeepStabilizer stabilizer;
  (void)stabilizer.stabilize(waveform({{70.0, 1.0}}), 70.0, 0.8);
  stabilizer.reset();
  const auto result =
      stabilizer.stabilize(waveform({{120.0, 1.0}}), 120.0, 0.8);
  EXPECT_TRUE(result.stability_valid);
  expect_near(result.display_bpm, 120.0);
  EXPECT_EQ(result.correction_reason, std::string("accepted_raw"));
}

}  // namespace

int main() {
  rejects_low_confidence();
  rejects_nonfinite_and_constant_waveforms();
  rejects_nonfinite_raw_bpm();
  corrects_half_and_double_only_with_spectral_support();
  rejects_unsupported_large_jump();
  allows_supported_large_jump_and_clears_history();
  rejected_value_does_not_pollute_history();
  displays_median_of_three_accepted_values();
  reset_discards_history();
  return test_support::finish();
}
