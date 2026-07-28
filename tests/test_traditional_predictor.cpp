#include "rppg_qnn/traditional_predictor.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "test_support.hpp"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kFps = 30.0;

std::vector<cv::Vec3d> python_reference_rgb(std::size_t count = 300U) {
  std::vector<cv::Vec3d> rgb;
  rgb.reserve(count);
  for (std::size_t index = 0; index < count; ++index) {
    const double time = static_cast<double>(index) / kFps;
    const double pulse = std::sin(2.0 * kPi * 1.2 * time);
    const double motion = std::sin(2.0 * kPi * 0.2 * time);
    rgb.emplace_back(120.0 * (1.0 + 0.04 * motion + 0.002 * pulse),
                     90.0 * (1.0 + 0.04 * motion + 0.010 * pulse),
                     70.0 * (1.0 + 0.04 * motion - 0.003 * pulse));
  }
  return rgb;
}

void expect_near(double actual, double expected, double tolerance) {
  EXPECT_TRUE(std::isfinite(actual));
  if (std::isfinite(actual) && std::abs(actual - expected) > tolerance) {
    std::cerr << "actual=" << actual << " expected=" << expected
              << " tolerance=" << tolerance << '\n';
  }
  EXPECT_TRUE(std::abs(actual - expected) <= tolerance);
}

void expect_all_finite(const std::vector<double>& values) {
  for (double value : values) {
    EXPECT_TRUE(std::isfinite(value));
  }
}

void pos_bvp_matches_python_reference() {
  const std::vector<double> actual = rppg_qnn::extract_traditional_bvp(
      python_reference_rgb(), rppg_qnn::TraditionalMethod::Pos);
  EXPECT_EQ(actual.size(), std::size_t{300});
  expect_all_finite(actual);

  const std::vector<std::pair<std::size_t, double>> selected{
      {0U, 0.015687221761180217},   {1U, 0.03313974802637247},
      {2U, 0.05650928652841992},    {47U, -0.7362807219928412},
      {48U, -0.5345373509339912},   {49U, -0.2923851557275933},
      {100U, -0.002739292859572977}, {150U, 0.004587713940371863},
      {200U, -0.002738691286683628}, {250U, 0.035339401863657416},
      {299U, -0.0336783656365066},
  };
  for (const auto& [index, expected] : selected) {
    expect_near(actual[index], expected, 2e-6);
  }

  const double sum = std::accumulate(actual.begin(), actual.end(), 0.0);
  const double square_sum = std::inner_product(
      actual.begin(), actual.end(), actual.begin(), 0.0);
  expect_near(sum, 0.0, 2e-8);
  expect_near(std::sqrt(square_sum), 12.018466457770694, 2e-5);
}

void chrom_bvp_matches_python_reference() {
  const std::vector<double> actual = rppg_qnn::extract_traditional_bvp(
      python_reference_rgb(), rppg_qnn::TraditionalMethod::Chrom);
  EXPECT_EQ(actual.size(), std::size_t{300});
  expect_all_finite(actual);

  const std::vector<std::pair<std::size_t, double>> selected{
      {0U, -0.003843479273584379},  {1U, -0.0038878049100979106},
      {2U, -0.004132126777380536}, {47U, 0.018520708363808536},
      {48U, 0.013482245354546929},  {49U, 0.007467998976492083},
      {100U, 0.003677762593106299}, {150U, 0.003238950089704869},
      {200U, 0.000569854354152024}, {250U, -0.001278384151621246},
      {299U, -0.003843479273584379},
  };
  for (const auto& [index, expected] : selected) {
    expect_near(actual[index], expected, 2e-9);
  }

  const double sum = std::accumulate(actual.begin(), actual.end(), 0.0);
  const double square_sum = std::inner_product(
      actual.begin(), actual.end(), actual.begin(), 0.0);
  expect_near(sum, 0.0, 2e-10);
  expect_near(std::sqrt(square_sum), 0.33643186946452214, 2e-8);
}

void selected_predictors_estimate_72_bpm() {
  for (const auto method : {rppg_qnn::TraditionalMethod::Pos,
                            rppg_qnn::TraditionalMethod::Chrom,
                            rppg_qnn::TraditionalMethod::Lgi}) {
    rppg_qnn::TraditionalPredictor predictor(method);
    const std::vector<cv::Vec3d> rgb = python_reference_rgb(601U);
    for (std::size_t index = 0; index < rgb.size(); ++index) {
      const cv::Vec3d& sample = rgb[index];
      predictor.add_sample(static_cast<double>(index) / kFps,
                           cv::Scalar(sample[2], sample[1], sample[0]));
    }
    const auto result = predictor.latest_result();
    EXPECT_TRUE(result.has_value());
    if (!result.has_value()) {
      continue;
    }
    EXPECT_EQ(result->method, rppg_qnn::traditional_method_name(method));
    EXPECT_TRUE(result->is_valid);
    EXPECT_TRUE(std::abs(result->bpm - 72.0) <= 0.01);
    EXPECT_TRUE(result->confidence >= 0.10);
    EXPECT_EQ(result->waveform.size(), std::size_t{300});
  }
}

void flat_rgb_is_invalid_without_fallback() {
  for (const auto method : {rppg_qnn::TraditionalMethod::Pos,
                            rppg_qnn::TraditionalMethod::Chrom}) {
    rppg_qnn::TraditionalPredictor predictor(method);
    for (int index = 0; index <= 360; ++index) {
      predictor.add_sample(static_cast<double>(index) / kFps,
                           cv::Scalar(70.0, 90.0, 120.0));
    }
    const auto result = predictor.latest_result();
    EXPECT_TRUE(result.has_value());
    if (result.has_value()) {
      EXPECT_EQ(result->method, rppg_qnn::traditional_method_name(method));
      EXPECT_TRUE(!result->is_valid);
      EXPECT_EQ(result->invalid_reason, "low_confidence");
    }
  }
}

void invalid_sampling_preserves_selected_method() {
  for (const auto method : {rppg_qnn::TraditionalMethod::Pos,
                            rppg_qnn::TraditionalMethod::Chrom}) {
    const std::string expected_method = rppg_qnn::traditional_method_name(method);

    rppg_qnn::TraditionalPredictor low_fps(method);
    for (int index = 0; index <= 120; ++index) {
      low_fps.add_sample(static_cast<double>(index) / 10.0,
                         cv::Scalar(70.0, 90.0, 120.0));
    }
    const auto low_fps_result = low_fps.latest_result();
    EXPECT_TRUE(low_fps_result.has_value());
    if (low_fps_result.has_value()) {
      EXPECT_EQ(low_fps_result->method, expected_method);
      EXPECT_EQ(low_fps_result->invalid_reason, "low_source_fps");
    }

    rppg_qnn::TraditionalPredictor nonmonotonic(method);
    for (int index = 0; index <= 300; ++index) {
      nonmonotonic.add_sample(static_cast<double>(index) / kFps,
                              cv::Scalar(70.0, 90.0, 120.0));
    }
    nonmonotonic.add_sample(9.0, cv::Scalar(70.0, 90.0, 120.0));
    const auto reset_result = nonmonotonic.latest_result();
    EXPECT_TRUE(reset_result.has_value());
    if (reset_result.has_value()) {
      EXPECT_EQ(reset_result->method, expected_method);
      EXPECT_EQ(reset_result->invalid_reason, "sampling");
    }
    EXPECT_EQ(nonmonotonic.buffered_count(), std::size_t{0});

    nonmonotonic.add_sample(
        std::numeric_limits<double>::quiet_NaN(),
        cv::Scalar(70.0, 90.0, 120.0));
    const auto nan_result = nonmonotonic.latest_result();
    EXPECT_TRUE(nan_result.has_value());
    if (nan_result.has_value()) {
      EXPECT_EQ(nan_result->method, expected_method);
      EXPECT_EQ(nan_result->invalid_reason, "sampling");
    }
  }
}

}  // namespace

int main() {
  pos_bvp_matches_python_reference();
  chrom_bvp_matches_python_reference();
  selected_predictors_estimate_72_bpm();
  flat_rgb_is_invalid_without_fallback();
  invalid_sampling_preserves_selected_method();
  EXPECT_EQ(rppg_qnn::traditional_method_from_string("lgi"),
            rppg_qnn::TraditionalMethod::Lgi);
  EXPECT_EQ(rppg_qnn::traditional_method_name(rppg_qnn::TraditionalMethod::Lgi),
            "LGI");
  return test_support::finish();
}
