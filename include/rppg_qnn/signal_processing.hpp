#pragma once

#include <vector>

namespace rppg_qnn {

struct IirFilter {
  std::vector<double> b;
  std::vector<double> a;
  std::vector<double> zi;
};

// Numerically stable mean using max-abs scaling to avoid overflow.
double stable_mean(const std::vector<double>& values);

// Subtract the mean in-place; falls back to zero on non-finite mean.
void demean(std::vector<double>* values);

// Forward IIR filter with state initialisation scaled by initial_scale.
std::vector<double> lfilter(const std::vector<double>& input,
                            const IirFilter& filter,
                            double initial_scale);

// Zero-phase forward-backward filter matching scipy.signal.filtfilt with
// odd-extension padding (edge = 3 * max(len(a), len(b))).
std::vector<double> filtfilt(const std::vector<double>& input,
                             const IirFilter& filter);

// Smoothness-prior detrending (Tarvainen et al.) with lambda=100.
// Solves (I + lambda^2 * D2'*D2) * trend = x and returns x - trend.
std::vector<double> smoothness_priors_detrend(const std::vector<double>& input);

}  // namespace rppg_qnn
