#include "rppg_qnn/efficientphys_postprocessor.hpp"

#include "test_support.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kExpectedSamples = 180U;

std::vector<float> load_fixture(const char* filename) {
  std::string path =
      std::string(RPPG_FIXTURE_DIR) + "/" + filename;
  std::ifstream file(path);
  if (!file) {
    std::cerr << "Cannot open fixture: " << path << '\n';
    return {};
  }
  std::vector<float> values;
  std::string line;
  while (std::getline(file, line)) {
    std::istringstream iss(line);
    double value;
    if (iss >> value) {
      values.push_back(static_cast<float>(value));
    }
  }
  return values;
}

double max_abs_error = 0.0;

void expect_near(double actual, double expected, double tolerance,
                 const char* file, int line) {
  const double error = std::abs(actual - expected);
  if (error >= tolerance) {
    std::cerr << file << ':' << line << ": expect_near failed: |"
              << actual << " - " << expected << "| = " << error
              << " >= " << tolerance << '\n';
    ++test_support::failures;
  }
  max_abs_error = std::max(max_abs_error, error);
}

#define EXPECT_NEAR(actual, expected, tolerance) \
  expect_near((actual), (expected), (tolerance), __FILE__, __LINE__)

double normalized_total_variation(const std::vector<float>& x) {
  double min_val = std::numeric_limits<double>::max();
  double max_val = std::numeric_limits<double>::lowest();
  for (float v : x) {
    min_val = std::min(min_val, static_cast<double>(v));
    max_val = std::max(max_val, static_cast<double>(v));
  }
  const double range = max_val - min_val;
  if (range < 1e-15) return 0.0;
  double tv = 0.0;
  for (std::size_t i = 1; i < x.size(); ++i) {
    tv += std::abs(static_cast<double>(x[i]) - static_cast<double>(x[i - 1]));
  }
  return tv / range;
}

std::size_t turning_points(const std::vector<float>& x) {
  if (x.size() < 3) return 0;
  std::size_t count = 0;
  for (std::size_t i = 1; i + 1 < x.size(); ++i) {
    const double prev = static_cast<double>(x[i - 1]);
    const double curr = static_cast<double>(x[i]);
    const double next = static_cast<double>(x[i + 1]);
    if ((curr > prev && curr > next) || (curr < prev && curr < next)) {
      ++count;
    }
  }
  return count;
}

// --- Test: C++ reconstruction matches Python frozen reference ---
void matches_python_frozen_reference() {
  const auto raw_diff =
      load_fixture("efficientphys_raw_diff_180.txt");
  EXPECT_EQ(raw_diff.size(), kExpectedSamples);

  const auto python_bvp =
      load_fixture("efficientphys_reconstructed_bvp_180.txt");
  EXPECT_EQ(python_bvp.size(), kExpectedSamples);

  const auto cpp_result =
      rppg_qnn::reconstruct_efficientphys_bvp(raw_diff);
  EXPECT_TRUE(cpp_result.is_valid);
  EXPECT_EQ(cpp_result.invalid_reason, std::string{});
  EXPECT_EQ(cpp_result.bvp.size(), kExpectedSamples);

  // Numerical match: each sample must be within 1e-4 of Python.
  for (std::size_t i = 0; i < kExpectedSamples; ++i) {
    EXPECT_NEAR(static_cast<double>(cpp_result.bvp[i]),
                static_cast<double>(python_bvp[i]), 1e-4);
  }
  std::cout << "EfficientPhys reconstruction max_abs_error=" << max_abs_error
            << '\n';
}

// --- Test: input must be exactly 180 finite floats ---
void rejects_invalid_input_size() {
  std::vector<float> too_short(179U, 0.1F);
  const auto result1 =
      rppg_qnn::reconstruct_efficientphys_bvp(too_short);
  EXPECT_TRUE(!result1.is_valid);
  EXPECT_EQ(result1.invalid_reason,
            std::string{"EFFICIENTPHYS_RAW_DIFF_INVALID"});

  std::vector<float> too_long(181U, 0.1F);
  const auto result2 =
      rppg_qnn::reconstruct_efficientphys_bvp(too_long);
  EXPECT_TRUE(!result2.is_valid);
  EXPECT_EQ(result2.invalid_reason,
            std::string{"EFFICIENTPHYS_RAW_DIFF_INVALID"});
}

// --- Test: non-finite input is rejected ---
void rejects_non_finite_input() {
  auto raw_diff =
      load_fixture("efficientphys_raw_diff_180.txt");
  EXPECT_EQ(raw_diff.size(), kExpectedSamples);

  raw_diff[42] = std::numeric_limits<float>::infinity();
  const auto result1 =
      rppg_qnn::reconstruct_efficientphys_bvp(raw_diff);
  EXPECT_TRUE(!result1.is_valid);

  raw_diff[42] = std::numeric_limits<float>::quiet_NaN();
  const auto result2 =
      rppg_qnn::reconstruct_efficientphys_bvp(raw_diff);
  EXPECT_TRUE(!result2.is_valid);
}

// --- Test: constant input produces valid but zero-mean output ---
void handles_constant_input() {
  const std::vector<float> constant(180U, 5.0F);
  const auto result =
      rppg_qnn::reconstruct_efficientphys_bvp(constant);
  EXPECT_TRUE(result.is_valid);
  EXPECT_EQ(result.bvp.size(), kExpectedSamples);

  // All values must be finite.
  for (float v : result.bvp) {
    EXPECT_TRUE(std::isfinite(v));
  }

  // Constant input → cumsum is linear ramp → detrend removes it →
  // band-pass removes DC → output should be near zero.
  double max_abs = 0.0;
  for (float v : result.bvp) {
    max_abs = std::max(max_abs, std::abs(static_cast<double>(v)));
  }
  EXPECT_TRUE(max_abs < 1e-6);
}

// --- Test: output is always 180 finite floats ---
void output_is_180_finite_floats() {
  auto raw_diff =
      load_fixture("efficientphys_raw_diff_180.txt");
  const auto result =
      rppg_qnn::reconstruct_efficientphys_bvp(raw_diff);
  EXPECT_TRUE(result.is_valid);
  EXPECT_EQ(result.bvp.size(), kExpectedSamples);
  for (float v : result.bvp) {
    EXPECT_TRUE(std::isfinite(v));
  }
}

// --- Test: DC offset and scaling of input do not change reconstructed shape ---
void dc_offset_and_scale_invariance() {
  const auto raw_diff =
      load_fixture("efficientphys_raw_diff_180.txt");

  const auto baseline =
      rppg_qnn::reconstruct_efficientphys_bvp(raw_diff);
  EXPECT_TRUE(baseline.is_valid);

  // Add DC offset
  std::vector<float> offset_diff(180U);
  for (std::size_t i = 0; i < 180U; ++i) {
    offset_diff[i] = raw_diff[i] + 3.0F;
  }
  const auto offset_result =
      rppg_qnn::reconstruct_efficientphys_bvp(offset_diff);
  EXPECT_TRUE(offset_result.is_valid);

  // DC offset in the diff adds a linear ramp to cumsum, which the
  // detrender and band-pass should remove. The reconstructed shape
  // should match up to numerical precision.
  for (std::size_t i = 0; i < 180U; ++i) {
    EXPECT_NEAR(static_cast<double>(offset_result.bvp[i]),
                static_cast<double>(baseline.bvp[i]), 1e-3);
  }

  // Scale invariance: scaling the input scales the output linearly.
  std::vector<float> scaled_diff(180U);
  for (std::size_t i = 0; i < 180U; ++i) {
    scaled_diff[i] = raw_diff[i] * 2.0F;
  }
  const auto scaled_result =
      rppg_qnn::reconstruct_efficientphys_bvp(scaled_diff);
  EXPECT_TRUE(scaled_result.is_valid);
  for (std::size_t i = 0; i < 180U; ++i) {
    EXPECT_NEAR(static_cast<double>(scaled_result.bvp[i]),
                2.0 * static_cast<double>(baseline.bvp[i]), 1e-3);
  }
}

// --- Test: reconstructed BVP has lower roughness than raw diff ---
void reconstruction_reduces_roughness() {
  const auto raw_diff =
      load_fixture("efficientphys_raw_diff_180.txt");

  const auto result =
      rppg_qnn::reconstruct_efficientphys_bvp(raw_diff);
  EXPECT_TRUE(result.is_valid);

  // Reconstructed should have significantly fewer turning points.
  const std::size_t raw_tp = turning_points(raw_diff);
  const std::size_t recon_tp = turning_points(result.bvp);
  EXPECT_TRUE(recon_tp < raw_tp);

  // Reconstructed should have lower normalized total variation.
  const double raw_tv = normalized_total_variation(raw_diff);
  const double recon_tv = normalized_total_variation(result.bvp);
  EXPECT_TRUE(recon_tv < raw_tv);
}

}  // namespace

int main() {
  matches_python_frozen_reference();
  rejects_invalid_input_size();
  rejects_non_finite_input();
  handles_constant_input();
  output_is_180_finite_floats();
  dc_offset_and_scale_invariance();
  reconstruction_reduces_roughness();
  return test_support::finish();
}
