#include "rppg_qnn/traditional_predictor.hpp"

#include "rppg_qnn/error.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace rppg_qnn {
namespace {

constexpr double kHistorySeconds = 30.0;
constexpr double kWindowSeconds = 10.0;
constexpr double kResampleFps = 30.0;
constexpr std::size_t kResampleCount = 300U;
constexpr double kMinimumSourceFps = 15.0;
constexpr double kMaximumGapSeconds = 0.75;
constexpr int kMinimumFrequencyBin = 7;
constexpr int kMaximumFrequencyBin = 30;
constexpr double kMinimumConfidence = 0.10;
constexpr double kMinimumBpm = 42.0;
constexpr double kMaximumBpm = 180.0;
constexpr double kPi = 3.14159265358979323846;
constexpr double kEpsilon = 1e-12;
constexpr double kPeakTieRelativeTolerance = 1e-4;
constexpr std::size_t kProjectionWindow = 48U;

bool finite_bgr(const cv::Scalar& mean_bgr) {
  return std::isfinite(mean_bgr[0]) && std::isfinite(mean_bgr[1]) &&
         std::isfinite(mean_bgr[2]);
}

bool finite_rgb(const cv::Vec3d& rgb) {
  return std::isfinite(rgb[0]) && std::isfinite(rgb[1]) &&
         std::isfinite(rgb[2]);
}

HeartRateResult invalid_result(TraditionalMethod method, const char* reason) {
  HeartRateResult result;
  result.method = traditional_method_name(method);
  result.invalid_reason = reason;
  result.backend = "cpu";
  return result;
}

bool finite_result(const HeartRateResult& result) {
  if (!std::isfinite(result.window_start_sec) ||
      !std::isfinite(result.window_end_sec) || !std::isfinite(result.bpm) ||
      !std::isfinite(result.confidence) || !std::isfinite(result.source_fps) ||
      !std::isfinite(result.max_frame_gap_sec) ||
      !std::isfinite(result.inference_ms)) {
    return false;
  }
  return std::all_of(result.waveform.begin(), result.waveform.end(),
                     [](float value) { return std::isfinite(value); });
}

HeartRateResult finite_failure_result(const HeartRateResult& result,
                                      TraditionalMethod method,
                                      const char* reason) {
  HeartRateResult safe = invalid_result(method, reason);
  safe.window_start_sec =
      std::isfinite(result.window_start_sec) ? result.window_start_sec : 0.0;
  safe.window_end_sec =
      std::isfinite(result.window_end_sec) ? result.window_end_sec : 0.0;
  safe.source_fps = std::isfinite(result.source_fps) ? result.source_fps : 0.0;
  safe.source_frame_count = result.source_frame_count;
  safe.max_frame_gap_sec = std::isfinite(result.max_frame_gap_sec)
                               ? result.max_frame_gap_sec
                               : 0.0;
  return safe;
}

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

double safe_denominator(double value) {
  return std::isfinite(value) && std::abs(value) >= kEpsilon ? value : 1.0;
}

double population_std(const std::vector<double>& values) {
  if (values.empty()) {
    return kEpsilon;
  }
  const double mean = stable_mean(values);
  if (!std::isfinite(mean)) {
    return kEpsilon;
  }
  double scale = 0.0;
  for (double value : values) {
    scale = std::max(scale, std::abs(value - mean));
  }
  if (!std::isfinite(scale) || scale < kEpsilon) {
    return kEpsilon;
  }
  double square_sum = 0.0;
  for (double value : values) {
    const double normalized = (value - mean) / scale;
    square_sum += normalized * normalized;
  }
  const double result = scale * std::sqrt(square_sum / static_cast<double>(values.size()));
  return std::isfinite(result) && result >= kEpsilon ? result : kEpsilon;
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

struct IirFilter {
  std::vector<double> b;
  std::vector<double> a;
  std::vector<double> zi;
};

const IirFilter& pos_filter() {
  static const IirFilter filter{
      {0.19359960593003395, 0.0, -0.19359960593003395},
      {1.0, -1.532373232577131, 0.6128007881399321},
      {-0.19359960593003334, -0.1935996059300343}};
  return filter;
}

const IirFilter& chrom_filter() {
  static const IirFilter filter{
      {0.004750523610980864, 0.0, -0.014251570832942593, 0.0,
       0.014251570832942593, 0.0, -0.004750523610980864},
      {1.0, -5.047460698732161, 10.80593920434056, -12.56877357562442,
       8.381625175580195, -3.0393274823186087, 0.46831211117171195},
      {-0.004750523611006747, -0.004750523610876101,
       0.009501047221786805, 0.00950104722211212,
       -0.004750523611047408, -0.004750523610968743}};
  return filter;
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
  return std::vector<double>(backward.begin() + static_cast<std::ptrdiff_t>(edge),
                             backward.begin() +
                                 static_cast<std::ptrdiff_t>(edge + input.size()));
}

std::array<double, 3> channel_means(const std::vector<cv::Vec3d>& rgb,
                                    std::size_t begin,
                                    std::size_t end) {
  std::array<double, 3> result{};
  for (std::size_t channel = 0; channel < result.size(); ++channel) {
    std::vector<double> values;
    values.reserve(end - begin);
    for (std::size_t index = begin; index < end; ++index) {
      values.push_back(rgb[index][static_cast<int>(channel)]);
    }
    result[channel] = safe_denominator(stable_mean(values));
  }
  return result;
}

std::vector<double> green_bvp(const std::vector<cv::Vec3d>& rgb) {
  const auto means = channel_means(rgb, 0U, rgb.size());
  std::vector<double> output;
  output.reserve(rgb.size());
  for (const cv::Vec3d& sample : rgb) {
    output.push_back(sample[1] / means[1]);
  }
  demean(&output);
  return output;
}

std::vector<double> smoothness_priors_detrend(const std::vector<double>& input) {
  const int length = static_cast<int>(input.size());
  if (length < 3) {
    std::vector<double> output = input;
    demean(&output);
    return output;
  }

  constexpr double lambda_squared = 10000.0;
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

std::vector<double> pos_bvp(const std::vector<cv::Vec3d>& rgb) {
  if (rgb.size() < kProjectionWindow) {
    return green_bvp(rgb);
  }
  std::vector<double> output(rgb.size(), 0.0);
  for (std::size_t end = kProjectionWindow; end < rgb.size(); ++end) {
    const std::size_t begin = end - kProjectionWindow;
    const auto means = channel_means(rgb, begin, end);
    std::vector<double> first;
    std::vector<double> second;
    first.reserve(kProjectionWindow);
    second.reserve(kProjectionWindow);
    for (std::size_t index = begin; index < end; ++index) {
      const double red = rgb[index][0] / means[0];
      const double green = rgb[index][1] / means[1];
      const double blue = rgb[index][2] / means[2];
      first.push_back(green - blue);
      second.push_back(-2.0 * red + green + blue);
    }
    const double alpha = population_std(first) / population_std(second);
    std::vector<double> segment(kProjectionWindow, 0.0);
    for (std::size_t index = 0; index < kProjectionWindow; ++index) {
      segment[index] = first[index] + alpha * second[index];
    }
    demean(&segment);
    for (std::size_t index = 0; index < kProjectionWindow; ++index) {
      output[begin + index] += segment[index];
    }
  }
  std::vector<double> filtered =
      filtfilt(smoothness_priors_detrend(output), pos_filter());
  demean(&filtered);
  return filtered;
}

std::vector<double> chrom_bvp(const std::vector<cv::Vec3d>& rgb) {
  if (rgb.size() < kProjectionWindow) {
    return green_bvp(rgb);
  }
  constexpr std::size_t half_window = kProjectionWindow / 2U;
  const std::size_t window_count = (rgb.size() - half_window) / half_window;
  std::vector<double> output(half_window * (window_count + 1U), 0.0);
  std::size_t begin = 0U;
  std::size_t middle = half_window;
  std::size_t end = kProjectionWindow;
  for (std::size_t window = 0; window < window_count; ++window) {
    const auto means = channel_means(rgb, begin, end);
    std::vector<double> x;
    std::vector<double> y;
    x.reserve(kProjectionWindow);
    y.reserve(kProjectionWindow);
    for (std::size_t index = begin; index < end; ++index) {
      const double red = rgb[index][0] / means[0];
      const double green = rgb[index][1] / means[1];
      const double blue = rgb[index][2] / means[2];
      x.push_back(3.0 * red - 2.0 * green);
      y.push_back(1.5 * red + green - 1.5 * blue);
    }
    const std::vector<double> filtered_x = filtfilt(x, chrom_filter());
    const std::vector<double> filtered_y = filtfilt(y, chrom_filter());
    const double alpha = population_std(filtered_x) / population_std(filtered_y);
    for (std::size_t index = 0; index < kProjectionWindow; ++index) {
      const double hann = 0.5 - 0.5 * std::cos(
          2.0 * kPi * static_cast<double>(index) /
          static_cast<double>(kProjectionWindow - 1U));
      const double value = (filtered_x[index] - alpha * filtered_y[index]) * hann;
      if (index < half_window) {
        output[begin + index] += value;
      } else {
        output[middle + index - half_window] = value;
      }
    }
    begin = middle;
    middle = begin + half_window;
    end = begin + kProjectionWindow;
  }
  output.resize(rgb.size(), 0.0);
  demean(&output);
  return output;
}

}  // namespace

TraditionalMethod traditional_method_from_string(const std::string& method) {
  if (method == "green" || method == "GREEN") {
    return TraditionalMethod::Green;
  }
  if (method == "pos" || method == "POS") {
    return TraditionalMethod::Pos;
  }
  if (method == "chrom" || method == "CHROM") {
    return TraditionalMethod::Chrom;
  }
  throw AppError(ErrorCode::ConfigInvalid,
                 "unsupported traditional method: " + method);
}

std::string traditional_method_name(TraditionalMethod method) {
  switch (method) {
    case TraditionalMethod::Green:
      return "GREEN";
    case TraditionalMethod::Pos:
      return "POS";
    case TraditionalMethod::Chrom:
      return "CHROM";
  }
  throw std::invalid_argument("unknown traditional method enum");
}

std::vector<double> extract_traditional_bvp(const std::vector<cv::Vec3d>& rgb,
                                            TraditionalMethod method) {
  if (!std::all_of(rgb.begin(), rgb.end(), finite_rgb)) {
    return std::vector<double>(rgb.size(), 0.0);
  }
  switch (method) {
    case TraditionalMethod::Green:
      return green_bvp(rgb);
    case TraditionalMethod::Pos:
      return pos_bvp(rgb);
    case TraditionalMethod::Chrom:
      return chrom_bvp(rgb);
  }
  return std::vector<double>(rgb.size(), 0.0);
}

TraditionalPredictor::TraditionalPredictor(TraditionalMethod method) : method_(method) {}

void TraditionalPredictor::add_sample(double timestamp_sec,
                                      const cv::Scalar& mean_bgr) {
  if (!std::isfinite(timestamp_sec) || !finite_bgr(mean_bgr) ||
      (!samples_.empty() && timestamp_sec <= samples_.back().timestamp_sec)) {
    reset();
    set_sampling_result();
    return;
  }

  samples_.push_back(
      {timestamp_sec, cv::Vec3d(mean_bgr[2], mean_bgr[1], mean_bgr[0])});
  while (!samples_.empty() &&
         samples_.front().timestamp_sec < timestamp_sec - kHistorySeconds) {
    samples_.pop_front();
  }

  if (samples_.empty() ||
      timestamp_sec - samples_.front().timestamp_sec < kWindowSeconds) {
    set_sampling_result();
    return;
  }

  if (!last_evaluation_timestamp_sec_.has_value() ||
      timestamp_sec - *last_evaluation_timestamp_sec_ >= 1.0) {
    evaluate();
  }
}

std::optional<HeartRateResult> TraditionalPredictor::latest_result() const {
  return latest_;
}

std::size_t TraditionalPredictor::buffered_count() const { return samples_.size(); }

double TraditionalPredictor::buffered_span_sec() const {
  if (samples_.size() < 2U) {
    return 0.0;
  }
  return samples_.back().timestamp_sec - samples_.front().timestamp_sec;
}

std::size_t TraditionalPredictor::evaluation_count() const {
  return evaluation_count_;
}

TraditionalMethod TraditionalPredictor::method() const { return method_; }

void TraditionalPredictor::reset() {
  samples_.clear();
  latest_.reset();
  last_evaluation_timestamp_sec_.reset();
  evaluation_count_ = 0;
}

void TraditionalPredictor::set_sampling_result() {
  latest_ = invalid_result(method_, "sampling");
}

void TraditionalPredictor::evaluate() {
  const double window_end = samples_.back().timestamp_sec;
  const double window_start = window_end - kWindowSeconds;

  HeartRateResult result = invalid_result(method_, "sampling");
  result.window_start_sec = window_start;
  result.window_end_sec = window_end;
  const auto lower = std::lower_bound(
      samples_.begin(), samples_.end(), window_start,
      [](const Sample& sample, double timestamp) {
        return sample.timestamp_sec < timestamp;
      });
  if (lower == samples_.end() ||
      (lower == samples_.begin() && lower->timestamp_sec > window_start)) {
    latest_ = result;
    last_evaluation_timestamp_sec_ = window_end;
    ++evaluation_count_;
    return;
  }

  const auto source_begin = lower == samples_.begin() ? lower : std::prev(lower);
  std::vector<Sample> source(source_begin, samples_.end());
  result.source_frame_count = source.size();
  if (source.size() < 2U) {
    latest_ = result;
    last_evaluation_timestamp_sec_ = window_end;
    ++evaluation_count_;
    return;
  }

  const double source_duration =
      source.back().timestamp_sec - source.front().timestamp_sec;
  if (!std::isfinite(source_duration) || source_duration <= 0.0) {
    latest_ = result;
    last_evaluation_timestamp_sec_ = window_end;
    ++evaluation_count_;
    return;
  }
  result.source_fps =
      static_cast<double>(source.size() - 1U) / source_duration;
  for (std::size_t index = 1; index < source.size(); ++index) {
    result.max_frame_gap_sec = std::max(
        result.max_frame_gap_sec,
        source[index].timestamp_sec - source[index - 1U].timestamp_sec);
  }

  if (result.source_fps < kMinimumSourceFps) {
    result.invalid_reason = "low_source_fps";
  } else if (result.max_frame_gap_sec > kMaximumGapSeconds) {
    result.invalid_reason = "capture_gap";
  } else {
    std::vector<cv::Vec3d> resampled;
    resampled.reserve(kResampleCount);
    std::size_t right_index = 1U;
    bool all_targets_covered = true;
    for (std::size_t index = 0; index < kResampleCount; ++index) {
      const double target_time =
          window_start + static_cast<double>(index) / kResampleFps;
      while (right_index < source.size() &&
             source[right_index].timestamp_sec < target_time) {
        ++right_index;
      }
      if (right_index == source.size()) {
        all_targets_covered = false;
        break;
      }
      const Sample& right = source[right_index];
      const Sample& left = source[right_index - 1U];
      if (left.timestamp_sec > target_time || right.timestamp_sec < target_time) {
        all_targets_covered = false;
        break;
      }
      const double span = right.timestamp_sec - left.timestamp_sec;
      if (span <= 0.0) {
        all_targets_covered = false;
        break;
      }
      const double fraction = (target_time - left.timestamp_sec) / span;
      cv::Vec3d interpolated;
      for (int channel = 0; channel < 3; ++channel) {
        const double pair_scale =
            std::max(std::abs(left.rgb[channel]), std::abs(right.rgb[channel]));
        if (!std::isfinite(pair_scale)) {
          all_targets_covered = false;
          break;
        }
        if (pair_scale == 0.0) {
          interpolated[channel] = 0.0;
          continue;
        }
        const double blended =
            (left.rgb[channel] / pair_scale) * (1.0 - fraction) +
            (right.rgb[channel] / pair_scale) * fraction;
        interpolated[channel] =
            pair_scale * std::clamp(blended, -1.0, 1.0);
      }
      if (!all_targets_covered || !finite_rgb(interpolated)) {
        all_targets_covered = false;
        break;
      }
      resampled.push_back(interpolated);
    }

    if (!all_targets_covered) {
      latest_ = result;
      last_evaluation_timestamp_sec_ = window_end;
      ++evaluation_count_;
      return;
    }

    std::vector<double> bvp = extract_traditional_bvp(resampled, method_);
    if (bvp.size() != kResampleCount ||
        !std::all_of(bvp.begin(), bvp.end(),
                     [](double value) { return std::isfinite(value); })) {
      latest_ = finite_failure_result(result, method_, "low_confidence");
      last_evaluation_timestamp_sec_ = window_end;
      ++evaluation_count_;
      return;
    }

    double bvp_scale = 0.0;
    for (double value : bvp) {
      bvp_scale = std::max(bvp_scale, std::abs(value));
    }
    if (!std::isfinite(bvp_scale)) {
      latest_ = finite_failure_result(result, method_, "low_confidence");
      last_evaluation_timestamp_sec_ = window_end;
      ++evaluation_count_;
      return;
    }
    if (bvp_scale > 0.0) {
      for (double& value : bvp) {
        value /= bvp_scale;
      }
    }

    double mean_time = 0.0;
    double mean_value = 0.0;
    for (std::size_t index = 0; index < bvp.size(); ++index) {
      mean_time += static_cast<double>(index) / kResampleFps;
      mean_value += bvp[index];
    }
    mean_time /= static_cast<double>(bvp.size());
    mean_value /= static_cast<double>(bvp.size());

    double time_variance = 0.0;
    double covariance = 0.0;
    for (std::size_t index = 0; index < bvp.size(); ++index) {
      const double time = static_cast<double>(index) / kResampleFps;
      time_variance += (time - mean_time) * (time - mean_time);
      covariance += (time - mean_time) * (bvp[index] - mean_value);
    }
    const double slope = time_variance > 0.0 ? covariance / time_variance : 0.0;

    std::vector<double> detrended;
    detrended.reserve(bvp.size());
    double maximum_absolute = 0.0;
    for (std::size_t index = 0; index < bvp.size(); ++index) {
      const double time = static_cast<double>(index) / kResampleFps;
      const double value = bvp[index] - mean_value - slope * (time - mean_time);
      detrended.push_back(value);
      maximum_absolute = std::max(maximum_absolute, std::abs(value));
    }
    std::vector<double> normalized_detrended;
    normalized_detrended.reserve(detrended.size());
    result.waveform.reserve(detrended.size());
    for (double value : detrended) {
      const double normalized =
          maximum_absolute > kEpsilon ? value / maximum_absolute : 0.0;
      normalized_detrended.push_back(normalized);
      result.waveform.push_back(static_cast<float>(normalized));
    }

    double total_power = 0.0;
    double peak_power = 0.0;
    double peak_frequency = 0.0;
    for (int frequency_bin = kMinimumFrequencyBin;
         frequency_bin <= kMaximumFrequencyBin; ++frequency_bin) {
      const double frequency = static_cast<double>(frequency_bin) / 10.0;
      double real = 0.0;
      double imaginary = 0.0;
      for (std::size_t index = 0; index < normalized_detrended.size(); ++index) {
        const double phase = -2.0 * kPi * frequency *
                             static_cast<double>(index) / kResampleFps;
        const double hann = 0.5 - 0.5 * std::cos(
            2.0 * kPi * static_cast<double>(index) /
            static_cast<double>(normalized_detrended.size() - 1U));
        const double weighted = normalized_detrended[index] * hann;
        real += weighted * std::cos(phase);
        imaginary += weighted * std::sin(phase);
      }
      const double power = real * real + imaginary * imaginary;
      total_power += power;
      const bool higher_power = power > peak_power;
      const bool near_equal_higher_frequency =
          frequency > peak_frequency && peak_power > 0.0 &&
          peak_power - power <= peak_power * kPeakTieRelativeTolerance;
      if (higher_power || near_equal_higher_frequency) {
        peak_power = power;
        peak_frequency = frequency;
      }
    }
    result.bpm = peak_frequency * 60.0;
    result.confidence =
        total_power > kEpsilon ? peak_power / total_power : 0.0;
    if (result.confidence >= kMinimumConfidence && result.bpm >= kMinimumBpm &&
        result.bpm <= kMaximumBpm) {
      result.is_valid = true;
      result.invalid_reason.clear();
    } else {
      result.invalid_reason = "low_confidence";
    }
  }

  if (!finite_result(result)) {
    latest_ = finite_failure_result(result, method_, "low_confidence");
    last_evaluation_timestamp_sec_ = window_end;
    ++evaluation_count_;
    return;
  }

  latest_ = result;
  last_evaluation_timestamp_sec_ = window_end;
  ++evaluation_count_;
}

}  // namespace rppg_qnn
