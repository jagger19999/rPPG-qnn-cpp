#include "rppg_qnn/result_sink.hpp"

#include "rppg_qnn/error.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <locale>
#include <sstream>
#include <system_error>
#include <utility>

namespace rppg_qnn {
namespace {

std::mutex terminal_mutex;

[[noreturn]] void output_failure(const std::string& message) {
  throw AppError(ErrorCode::OutputWriteFailed, message);
}

std::ostringstream classic_output() {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  return output;
}

void append_replacement_character(std::string& output) {
  output += "\xef\xbf\xbd";
}

bool is_continuation(unsigned char value) { return (value & 0xc0U) == 0x80U; }

std::string sanitize_utf8(const std::string& value) {
  std::string output;
  output.reserve(value.size());
  for (std::size_t index = 0; index < value.size();) {
    const unsigned char first = static_cast<unsigned char>(value[index]);
    if (first <= 0x7fU) {
      output += static_cast<char>(first);
      ++index;
      continue;
    }
    const auto byte_at = [&value, index](std::size_t offset) {
      return static_cast<unsigned char>(value[index + offset]);
    };
    if (first >= 0xc2U && first <= 0xdfU && index + 1 < value.size() &&
        is_continuation(byte_at(1))) {
      output.append(value, index, 2);
      index += 2;
    } else if (first >= 0xe0U && first <= 0xefU && index + 2 < value.size() &&
               is_continuation(byte_at(1)) && is_continuation(byte_at(2)) &&
               !(first == 0xe0U && byte_at(1) < 0xa0U) &&
               !(first == 0xedU && byte_at(1) > 0x9fU)) {
      output.append(value, index, 3);
      index += 3;
    } else if (first >= 0xf0U && first <= 0xf4U && index + 3 < value.size() &&
               is_continuation(byte_at(1)) && is_continuation(byte_at(2)) &&
               is_continuation(byte_at(3)) && !(first == 0xf0U && byte_at(1) < 0x90U) &&
               !(first == 0xf4U && byte_at(1) > 0x8fU)) {
      output.append(value, index, 4);
      index += 4;
    } else {
      append_replacement_character(output);
      ++index;
    }
  }
  return output;
}

std::string json_escape(const std::string& value) {
  std::ostringstream output = classic_output();
  for (unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20U) {
          output << "\\u00" << std::uppercase << std::hex << std::setw(2)
                 << std::setfill('0') << static_cast<unsigned int>(character)
                 << std::nouppercase << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
    }
  }
  return output.str();
}

std::string json_string(const std::string& value) {
  return "\"" + json_escape(sanitize_utf8(value)) + "\"";
}

std::string json_number(double value) {
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream output = classic_output();
  output << std::setprecision(17) << value;
  return output.str();
}

std::string csv_field(const std::string& original_value) {
  const std::string value = sanitize_utf8(original_value);
  if (value.find_first_of(",\"\r\n") == std::string::npos) {
    return value;
  }
  std::string escaped;
  escaped.reserve(value.size() + 2);
  escaped += '"';
  for (char character : value) {
    if (character == '"') {
      escaped += '"';
    }
    escaped += character;
  }
  escaped += '"';
  return escaped;
}

std::string csv_number(double value) {
  return std::isfinite(value) ? json_number(value) : "";
}

std::string json_bool(bool value) { return value ? "true" : "false"; }

}  // namespace

ResultSink::ResultSink(std::filesystem::path output_dir) : output_dir_(std::move(output_dir)) {
  std::error_code error;
  std::filesystem::create_directories(output_dir_, error);
  if (error) {
    output_failure("could not create output directory: " + error.message());
  }
  events_.open(output_dir_ / "events.jsonl", std::ios::out | std::ios::trunc | std::ios::binary);
  heart_rate_.open(output_dir_ / "heart_rate.csv",
                   std::ios::out | std::ios::trunc | std::ios::binary);
  if (!events_.is_open() || !heart_rate_.is_open()) {
    events_.close();
    heart_rate_.close();
    output_failure("could not open result output files");
  }
  events_.imbue(std::locale::classic());
  heart_rate_.imbue(std::locale::classic());
  heart_rate_ << "schema_version,method,window_start_sec,window_end_sec,bpm,confidence,"
                 "is_valid,invalid_reason,source_fps,source_frame_count,max_frame_gap_sec,"
                 "window_materialization_ms,preprocess_ms,runtime_ms,postprocess_ms,"
                 "inference_ms,backend,model_sha256\r\n";
  if (!heart_rate_) {
    fail_permanently("could not write heart-rate CSV header");
  }
}

ResultSink::~ResultSink() noexcept {
  try {
    close(1);
  } catch (...) {
  }
}

[[noreturn]] void ResultSink::fail_permanently(std::string message) {
  if (state_ != State::Failed) {
    failed_message_ = std::move(message);
    state_ = State::Failed;
  }
  output_failure(failed_message_);
}

void ResultSink::ensure_open_for_publish() const {
  if (state_ == State::Failed) {
    output_failure(failed_message_);
  }
  if (state_ != State::Open) {
    output_failure("result sink is closed");
  }
}

void ResultSink::publish(const PreflightResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_open_for_publish();
  events_ << "{\"schema_version\":" << result.schema_version
          << ",\"event_type\":\"preflight_result\""
          << ",\"qnn_gpu_available\":" << json_bool(result.qnn_gpu_available)
          << ",\"opencl_available\":" << json_bool(result.opencl_available)
          << ",\"qnn_gpu_library\":" << json_string(result.qnn_gpu_library)
          << ",\"opencl_library\":" << json_string(result.opencl_library)
          << ",\"error\":" << json_string(result.error) << "}\n";
  if (!events_) {
    fail_permanently("could not write preflight result");
  }
  ++preflight_count_;
}

void ResultSink::publish_runtime_error(const std::string& error_code,
                                       const std::string& message) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_open_for_publish();
  events_ << "{\"schema_version\":1,\"event_type\":\"runtime_error\""
          << ",\"error_code\":" << json_string(error_code)
          << ",\"message\":" << json_string(message) << "}\n";
  events_.flush();
  if (!events_) {
    fail_permanently("could not write runtime error");
  }
  ++runtime_error_count_;
}

void ResultSink::publish(const FrameHealth& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_open_for_publish();
  if (!std::isfinite(result.timestamp_sec) ||
      (has_frame_timestamp_ && result.timestamp_sec < last_frame_timestamp_sec_) ||
      (has_frame_timestamp_ && result.timestamp_sec - last_frame_timestamp_sec_ < 1.0)) {
    return;
  }
  events_ << "{\"schema_version\":" << result.schema_version
          << ",\"event_type\":\"frame_health\""
          << ",\"frame_id\":" << result.frame_id
          << ",\"timestamp_sec\":" << json_number(result.timestamp_sec)
          << ",\"capture_fps\":" << json_number(result.capture_fps)
          << ",\"face_found\":" << json_bool(result.face_found)
          << ",\"face_confidence\":" << json_number(result.face_confidence)
          << ",\"status\":" << json_string(result.status) << "}\n";
  if (!events_) {
    fail_permanently("could not write frame-health result");
  }
  has_frame_timestamp_ = true;
  last_frame_timestamp_sec_ = result.timestamp_sec;
  ++frame_health_count_;
  std::ostringstream terminal = classic_output();
  terminal << "frame_health frame_id=" << result.frame_id
           << " timestamp_sec=" << json_number(result.timestamp_sec)
           << " capture_fps=" << json_number(result.capture_fps)
           << " face_found=" << json_bool(result.face_found)
           << " face_confidence=" << json_number(result.face_confidence)
           << " status=" << sanitize_utf8(result.status) << '\n';
  std::lock_guard<std::mutex> terminal_lock(terminal_mutex);
  std::cout << terminal.str();
}

void ResultSink::publish(const HeartRateResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  ensure_open_for_publish();
  events_ << "{\"schema_version\":" << result.schema_version
          << ",\"event_type\":\"heart_rate_result\""
          << ",\"method\":" << json_string(result.method)
          << ",\"window_start_sec\":" << json_number(result.window_start_sec)
          << ",\"window_end_sec\":" << json_number(result.window_end_sec)
          << ",\"bpm\":" << json_number(result.bpm)
          << ",\"confidence\":" << json_number(result.confidence)
          << ",\"is_valid\":" << json_bool(result.is_valid)
          << ",\"invalid_reason\":" << json_string(result.invalid_reason)
          << ",\"source_fps\":" << json_number(result.source_fps)
          << ",\"source_frame_count\":" << result.source_frame_count
          << ",\"max_frame_gap_sec\":" << json_number(result.max_frame_gap_sec)
          << ",\"window_materialization_ms\":"
          << json_number(result.window_materialization_ms)
          << ",\"preprocess_ms\":" << json_number(result.preprocess_ms)
          << ",\"runtime_ms\":" << json_number(result.runtime_ms)
          << ",\"postprocess_ms\":" << json_number(result.postprocess_ms)
          << ",\"inference_ms\":" << json_number(result.inference_ms)
          << ",\"backend\":" << json_string(result.backend)
          << ",\"model_sha256\":" << json_string(result.model_sha256) << "}\n";
  heart_rate_ << result.schema_version << ',' << csv_field(result.method) << ','
              << csv_number(result.window_start_sec) << ','
              << csv_number(result.window_end_sec) << ',' << csv_number(result.bpm) << ','
              << csv_number(result.confidence) << ',' << json_bool(result.is_valid) << ','
              << csv_field(result.invalid_reason) << ',' << csv_number(result.source_fps) << ','
              << result.source_frame_count << ',' << csv_number(result.max_frame_gap_sec) << ','
              << csv_number(result.window_materialization_ms) << ','
              << csv_number(result.preprocess_ms) << ',' << csv_number(result.runtime_ms) << ','
              << csv_number(result.postprocess_ms) << ','
              << csv_number(result.inference_ms) << ',' << csv_field(result.backend) << ','
              << csv_field(result.model_sha256) << "\r\n";
  events_.flush();
  heart_rate_.flush();
  if (!events_) {
    fail_permanently("could not flush heart-rate event");
  }
  if (!heart_rate_) {
    fail_permanently("could not flush heart-rate CSV");
  }
  ++heart_rate_count_;
  if (result.is_valid) {
    ++valid_heart_rate_count_;
  } else {
    ++invalid_heart_rate_count_;
  }
  std::ostringstream terminal = classic_output();
  terminal << "heart_rate bpm=" << json_number(result.bpm)
           << " confidence=" << json_number(result.confidence)
           << " valid=" << json_bool(result.is_valid) << '\n';
  std::lock_guard<std::mutex> terminal_lock(terminal_mutex);
  std::cout << terminal.str();
}

void ResultSink::close(int exit_code) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (state_ == State::Closed) {
    return;
  }
  if (state_ == State::Failed) {
    output_failure(failed_message_);
  }
  if (state_ == State::Open) {
    events_.flush();
    heart_rate_.flush();
    if (!events_) {
      fail_permanently("could not flush events output");
    }
    if (!heart_rate_) {
      fail_permanently("could not flush heart-rate output");
    }
    events_.close();
    heart_rate_.close();
    if (!events_) {
      fail_permanently("could not close events output");
    }
    if (!heart_rate_) {
      fail_permanently("could not close heart-rate output");
    }
    summary_exit_code_ = exit_code;
    state_ = State::SummaryPending;
  }

  const std::filesystem::path temporary = output_dir_ / "session_summary.json.tmp";
  const std::filesystem::path summary_path = output_dir_ / "session_summary.json";
  std::ofstream summary(temporary, std::ios::out | std::ios::trunc | std::ios::binary);
  summary.imbue(std::locale::classic());
  if (!summary.is_open()) {
    output_failure("could not open temporary session summary");
  }
  summary << "{\"schema_version\":1,\"event_type\":\"session_end\",\"exit_code\":"
          << summary_exit_code_ << ",\"preflight_count\":" << preflight_count_
          << ",\"frame_health_count\":" << frame_health_count_
          << ",\"runtime_error_count\":" << runtime_error_count_
          << ",\"heart_rate_count\":" << heart_rate_count_
          << ",\"valid_heart_rate_count\":" << valid_heart_rate_count_
          << ",\"invalid_heart_rate_count\":" << invalid_heart_rate_count_ << '}';
  summary.flush();
  if (!summary) {
    summary.close();
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    output_failure("could not write temporary session summary");
  }
  summary.close();
  if (!summary) {
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    output_failure("could not close temporary session summary");
  }
  std::error_code rename_error;
  std::filesystem::rename(temporary, summary_path, rename_error);
  if (rename_error) {
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    output_failure("could not publish session summary: " + rename_error.message());
  }
  state_ = State::Closed;
}

}  // namespace rppg_qnn
