#include "rppg_qnn/result_sink.hpp"

#include "rppg_qnn/error.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <system_error>
#include <utility>

namespace rppg_qnn {
namespace {

[[noreturn]] void output_failure(const std::string& message) {
  throw AppError(ErrorCode::OutputWriteFailed, message);
}

std::string json_escape(const std::string& value) {
  std::ostringstream output;
  for (unsigned char character : value) {
    switch (character) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (character < 0x20U) {
          output << "\\u00" << std::uppercase << std::hex << std::setw(2)
                 << std::setfill('0') << static_cast<unsigned int>(character)
                 << std::nouppercase << std::dec << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
        break;
    }
  }
  return output.str();
}

std::string json_string(const std::string& value) {
  return "\"" + json_escape(value) + "\"";
}

std::string json_number(double value) {
  if (!std::isfinite(value)) {
    return "null";
  }
  std::ostringstream output;
  output << std::setprecision(17) << value;
  return output.str();
}

std::string csv_field(const std::string& value) {
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

void check_stream(const std::ios& stream, const char* operation) {
  if (!stream) {
    output_failure(std::string("could not ") + operation);
  }
}

}  // namespace

ResultSink::ResultSink(std::filesystem::path output_dir) : output_dir_(std::move(output_dir)) {
  std::error_code error;
  std::filesystem::create_directories(output_dir_, error);
  if (error) {
    output_failure("could not create output directory: " + error.message());
  }

  events_.open(output_dir_ / "events.jsonl", std::ios::out | std::ios::trunc);
  heart_rate_.open(output_dir_ / "heart_rate.csv", std::ios::out | std::ios::trunc);
  if (!events_.is_open() || !heart_rate_.is_open()) {
    output_failure("could not open result output files");
  }
  heart_rate_ << "schema_version,method,window_start_sec,window_end_sec,bpm,confidence,"
                 "is_valid,invalid_reason,source_fps,source_frame_count,max_frame_gap_sec,"
                 "inference_ms,backend,model_sha256\r\n";
  check_stream(heart_rate_, "write heart-rate CSV header");
}

ResultSink::~ResultSink() noexcept {
  try {
    close(1);
  } catch (...) {
  }
}

void ResultSink::publish(const PreflightResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    output_failure("result sink is closed");
  }

  events_ << "{\"schema_version\":" << result.schema_version
          << ",\"event_type\":\"preflight_result\""
          << ",\"qnn_gpu_available\":" << json_bool(result.qnn_gpu_available)
          << ",\"opencl_available\":" << json_bool(result.opencl_available)
          << ",\"qnn_gpu_library\":" << json_string(result.qnn_gpu_library)
          << ",\"opencl_library\":" << json_string(result.opencl_library)
          << ",\"error\":" << json_string(result.error) << "}\n";
  check_stream(events_, "write preflight result");
  ++preflight_count_;
}

void ResultSink::publish(const FrameHealth& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    output_failure("result sink is closed");
  }
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
  check_stream(events_, "write frame-health result");
  has_frame_timestamp_ = true;
  last_frame_timestamp_sec_ = result.timestamp_sec;
  ++frame_health_count_;
}

void ResultSink::publish(const HeartRateResult& result) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    output_failure("result sink is closed");
  }

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
          << ",\"inference_ms\":" << json_number(result.inference_ms)
          << ",\"backend\":" << json_string(result.backend)
          << ",\"model_sha256\":" << json_string(result.model_sha256) << "}\n";

  heart_rate_ << result.schema_version << ',' << csv_field(result.method) << ','
              << csv_number(result.window_start_sec) << ','
              << csv_number(result.window_end_sec) << ',' << csv_number(result.bpm) << ','
              << csv_number(result.confidence) << ',' << json_bool(result.is_valid) << ','
              << csv_field(result.invalid_reason) << ',' << csv_number(result.source_fps) << ','
              << result.source_frame_count << ',' << csv_number(result.max_frame_gap_sec) << ','
              << csv_number(result.inference_ms) << ',' << csv_field(result.backend) << ','
              << csv_field(result.model_sha256) << "\r\n";
  events_.flush();
  heart_rate_.flush();
  check_stream(events_, "flush heart-rate event");
  check_stream(heart_rate_, "flush heart-rate CSV");

  ++heart_rate_count_;
  if (result.is_valid) {
    ++valid_heart_rate_count_;
  } else {
    ++invalid_heart_rate_count_;
  }
  std::cout << "heart_rate bpm=" << json_number(result.bpm)
            << " confidence=" << json_number(result.confidence)
            << " valid=" << json_bool(result.is_valid) << '\n';
}

void ResultSink::close(int exit_code) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (closed_) {
    return;
  }

  events_.flush();
  heart_rate_.flush();
  check_stream(events_, "flush events output");
  check_stream(heart_rate_, "flush heart-rate output");
  events_.close();
  heart_rate_.close();
  check_stream(events_, "close events output");
  check_stream(heart_rate_, "close heart-rate output");

  const std::filesystem::path temporary = output_dir_ / "session_summary.json.tmp";
  const std::filesystem::path summary_path = output_dir_ / "session_summary.json";
  std::ofstream summary(temporary, std::ios::out | std::ios::trunc);
  if (!summary.is_open()) {
    output_failure("open temporary session summary");
  }
  summary << "{\"schema_version\":1,\"event_type\":\"session_end\",\"exit_code\":"
          << exit_code << ",\"preflight_count\":" << preflight_count_
          << ",\"frame_health_count\":" << frame_health_count_
          << ",\"heart_rate_count\":" << heart_rate_count_
          << ",\"valid_heart_rate_count\":" << valid_heart_rate_count_
          << ",\"invalid_heart_rate_count\":" << invalid_heart_rate_count_ << '}';
  summary.flush();
  if (!summary) {
    summary.close();
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    output_failure("write temporary session summary");
  }
  summary.close();
  if (!summary) {
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    output_failure("close temporary session summary");
  }

  std::error_code rename_error;
  std::filesystem::rename(temporary, summary_path, rename_error);
  if (rename_error) {
    std::error_code cleanup_error;
    std::filesystem::remove(temporary, cleanup_error);
    output_failure("publish session summary: " + rename_error.message());
  }
  closed_ = true;
}

}  // namespace rppg_qnn
