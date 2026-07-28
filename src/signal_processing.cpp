#include "rppg_qnn/signal_processing.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>

#include <opencv2/core.hpp>

namespace rppg_qnn {

double stable_mean(const std::vector<double>& values) {
  if (values.empty()) {
    return 0.0;
  }
  double scale = 0.0;
  for (double value : values) {
    if (!std::isfinite(value)) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    scale = std::max(scale, std::abs(value));
  }
  if (scale == 0.0) {
    return 0.0;
  }
  double scaled_sum = 0.0;
  for (double value : values) {
    scaled_sum += value / scale;
  }
  return scale * scaled_sum / static_cast<double>(values.size());
}

void demean(std::vector<double>* values) {
  const double mean = stable_mean(*values);
  if (!std::isfinite(mean)) {
    std::fill(values->begin(), values->end(), 0.0);
    return;
  }
  for (double& value : *values) {
    value -= mean;
  }
}

std::vector<double> lfilter(const std::vector<double>& input,
                            const IirFilter& filter,
                            double initial_scale) {
  const std::size_t order = filter.a.size() - 1U;
  std::vector<double> state(order, 0.0);
  for (std::size_t index = 0; index < order; ++index) {
    state[index] = filter.zi[index] * initial_scale;
  }
  std::vector<double> output;
  output.reserve(input.size());
  for (double sample : input) {
    const double value = filter.b[0] * sample + state[0];
    for (std::size_t index = 0; index + 1U < order; ++index) {
      state[index] = filter.b[index + 1U] * sample + state[index + 1U] -
                     filter.a[index + 1U] * value;
    }
    state[order - 1U] =
        filter.b[order] * sample - filter.a[order] * value;
    output.push_back(value);
  }
  return output;
}

std::vector<double> filtfilt(const std::vector<double>& input,
                             const IirFilter& filter) {
  const std::size_t edge = 3U * std::max(filter.a.size(), filter.b.size());
  if (input.size() <= edge || filter.a.size() != filter.b.size() ||
      filter.zi.size() + 1U != filter.a.size()) {
    std::vector<double> fallback = input;
    demean(&fallback);
    return fallback;
  }

  std::vector<double> extended;
  extended.reserve(input.size() + 2U * edge);
  for (std::size_t index = edge; index > 0U; --index) {
    extended.push_back(2.0 * input.front() - input[index]);
  }
  extended.insert(extended.end(), input.begin(), input.end());
  for (std::size_t index = 0; index < edge; ++index) {
    extended.push_back(2.0 * input.back() - input[input.size() - 2U - index]);
  }

  std::vector<double> forward = lfilter(extended, filter, extended.front());
  std::reverse(forward.begin(), forward.end());
  std::vector<double> backward = lfilter(forward, filter, forward.front());
  std::reverse(backward.begin(), backward.end());
  return std::vector<double>(
      backward.begin() + static_cast<std::ptrdiff_t>(edge),
      backward.begin() +
          static_cast<std::ptrdiff_t>(edge + input.size()));
}

std::vector<double> smoothness_priors_detrend(const std::vector<double>& input) {
  const int length = static_cast<int>(input.size());
  if (length < 3) {
    std::vector<double> output = input;
    demean(&output);
    return output;
  }

  constexpr double lambda_squared = 10000.0;  // lambda = 100
  cv::Mat matrix = cv::Mat::eye(length, length, CV_64F);
  for (int row = 0; row < length - 2; ++row) {
    const std::array<int, 3> columns{row, row + 1, row + 2};
    const std::array<double, 3> coefficients{1.0, -2.0, 1.0};
    for (std::size_t left = 0; left < columns.size(); ++left) {
      for (std::size_t right = 0; right < columns.size(); ++right) {
        matrix.at<double>(columns[left], columns[right]) +=
            lambda_squared * coefficients[left] * coefficients[right];
      }
    }
  }

  cv::Mat source(length, 1, CV_64F);
  for (int index = 0; index < length; ++index) {
    source.at<double>(index, 0) = input[static_cast<std::size_t>(index)];
  }
  cv::Mat smooth;
  if (!cv::solve(matrix, source, smooth, cv::DECOMP_CHOLESKY)) {
    return std::vector<double>(input.size(), 0.0);
  }

  std::vector<double> output(input.size(), 0.0);
  for (int index = 0; index < length; ++index) {
    output[static_cast<std::size_t>(index)] =
        input[static_cast<std::size_t>(index)] - smooth.at<double>(index, 0);
  }
  return output;
}

}  // namespace rppg_qnn
