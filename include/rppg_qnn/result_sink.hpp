#pragma once

#include "rppg_qnn/contracts.hpp"

#include <filesystem>
#include <fstream>
#include <mutex>
#include <string>

namespace rppg_qnn {

class ResultSink {
 public:
  explicit ResultSink(std::filesystem::path output_dir);

  // The destructor makes a best-effort close with exit code 1 and never throws.
  // Call close explicitly to observe output failures.
  ~ResultSink() noexcept;

  ResultSink(const ResultSink&) = delete;
  ResultSink& operator=(const ResultSink&) = delete;

  void publish(const PreflightResult& result);
  void publish_runtime_error(const std::string& error_code, const std::string& message);
  // Frame-health events are emitted for the first finite timestamp and then at
  // most once per elapsed second. Non-finite and backwards timestamps are
  // discarded so they cannot defeat the throttle.
  void publish(const FrameHealth& result);
  void publish(const HeartRateResult& result);
  void close(int exit_code);

 private:
  enum class State { Open, SummaryPending, Closed, Failed };

  [[noreturn]] void fail_permanently(std::string message);
  void ensure_open_for_publish() const;

  std::filesystem::path output_dir_;
  std::ofstream events_;
  std::ofstream heart_rate_;
  std::mutex mutex_;
  State state_{State::Open};
  int summary_exit_code_{0};
  std::string failed_message_;
  bool has_frame_timestamp_{false};
  double last_frame_timestamp_sec_{0.0};
  std::size_t preflight_count_{0};
  std::size_t runtime_error_count_{0};
  std::size_t frame_health_count_{0};
  std::size_t heart_rate_count_{0};
  std::size_t valid_heart_rate_count_{0};
  std::size_t invalid_heart_rate_count_{0};
};

}  // namespace rppg_qnn
