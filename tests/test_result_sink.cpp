#include "rppg_qnn/contracts.hpp"
#include "rppg_qnn/result_sink.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "test_support.hpp"

namespace {

class ScopedDirectory {
 public:
  ScopedDirectory()
      : path_(std::filesystem::temp_directory_path() /
              ("rppg_qnn_result_sink_" + std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()))) {}

  ~ScopedDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }

  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

class ScopedCoutCapture {
 public:
  ScopedCoutCapture() : previous_(std::cout.rdbuf(stream_.rdbuf())) {}
  ~ScopedCoutCapture() { std::cout.rdbuf(previous_); }
  std::string str() const { return stream_.str(); }

 private:
  std::ostringstream stream_;
  std::streambuf* previous_;
};

class ScopedGlobalLocale {
 public:
  explicit ScopedGlobalLocale(const std::locale& replacement) : previous_(std::locale()) {
    std::locale::global(replacement);
  }
  ~ScopedGlobalLocale() { std::locale::global(previous_); }

 private:
  std::locale previous_;
};

class CommaNumpunct final : public std::numpunct<char> {
 protected:
  char do_decimal_point() const override { return ','; }
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<std::string> json_lines(const std::string& contents) {
  std::vector<std::string> lines;
  std::istringstream input(contents);
  for (std::string line; std::getline(input, line);) {
    lines.push_back(line);
  }
  return lines;
}

bool is_valid_utf8(const std::string& value) {
  for (std::size_t index = 0; index < value.size();) {
    const unsigned char first = static_cast<unsigned char>(value[index++]);
    if (first <= 0x7fU) {
      continue;
    }
    int remaining = 0;
    unsigned int code_point = 0;
    if (first >= 0xc2U && first <= 0xdfU) {
      remaining = 1;
      code_point = first & 0x1fU;
    } else if (first >= 0xe0U && first <= 0xefU) {
      remaining = 2;
      code_point = first & 0x0fU;
    } else if (first >= 0xf0U && first <= 0xf4U) {
      remaining = 3;
      code_point = first & 0x07U;
    } else {
      return false;
    }
    if (index + static_cast<std::size_t>(remaining) > value.size()) {
      return false;
    }
    for (int count = 0; count < remaining; ++count) {
      const unsigned char continuation = static_cast<unsigned char>(value[index++]);
      if ((continuation & 0xc0U) != 0x80U) {
        return false;
      }
      code_point = (code_point << 6U) | (continuation & 0x3fU);
    }
    if ((remaining == 1 && code_point < 0x80U) ||
        (remaining == 2 && code_point < 0x800U) ||
        (remaining == 3 && code_point < 0x10000U) ||
        (code_point >= 0xd800U && code_point <= 0xdfffU) || code_point > 0x10ffffU) {
      return false;
    }
  }
  return true;
}

class JsonParser {
 public:
  explicit JsonParser(const std::string& value) : value_(value) {}

  bool parse_object_document() {
    return parse_object() && position_ == value_.size();
  }

 private:
  bool parse_object() {
    if (!consume('{')) {
      return false;
    }
    if (consume('}')) {
      return true;
    }
    do {
      if (!parse_string() || !consume(':') || !parse_value()) {
        return false;
      }
    } while (consume(','));
    return consume('}');
  }

  bool parse_value() {
    if (position_ == value_.size()) {
      return false;
    }
    switch (value_[position_]) {
      case '"':
        return parse_string();
      case '{':
        return parse_object();
      case 't':
        return consume_literal("true");
      case 'f':
        return consume_literal("false");
      case 'n':
        return consume_literal("null");
      default:
        return parse_number();
    }
  }

  bool parse_string() {
    if (!consume('"')) {
      return false;
    }
    while (position_ < value_.size()) {
      const unsigned char character = static_cast<unsigned char>(value_[position_++]);
      if (character == '"') {
        return true;
      }
      if (character < 0x20U) {
        return false;
      }
      if (character != '\\') {
        continue;
      }
      if (position_ == value_.size()) {
        return false;
      }
      const char escape = value_[position_++];
      if (escape == '"' || escape == '\\' || escape == '/' || escape == 'b' ||
          escape == 'f' || escape == 'n' || escape == 'r' || escape == 't') {
        continue;
      }
      if (escape != 'u' || position_ + 4 > value_.size()) {
        return false;
      }
      for (int count = 0; count < 4; ++count) {
        const char hexadecimal = value_[position_++];
        if (!((hexadecimal >= '0' && hexadecimal <= '9') ||
              (hexadecimal >= 'a' && hexadecimal <= 'f') ||
              (hexadecimal >= 'A' && hexadecimal <= 'F'))) {
          return false;
        }
      }
    }
    return false;
  }

  bool parse_number() {
    const std::size_t start = position_;
    consume('-');
    if (consume('0')) {
      // JSON numbers cannot have a leading zero followed by another digit.
    } else if (position_ < value_.size() && value_[position_] >= '1' &&
               value_[position_] <= '9') {
      do {
        ++position_;
      } while (position_ < value_.size() && value_[position_] >= '0' &&
               value_[position_] <= '9');
    } else {
      position_ = start;
      return false;
    }
    if (consume('.')) {
      const std::size_t fraction_start = position_;
      while (position_ < value_.size() && value_[position_] >= '0' &&
             value_[position_] <= '9') {
        ++position_;
      }
      if (fraction_start == position_) {
        return false;
      }
    }
    if (position_ < value_.size() && (value_[position_] == 'e' || value_[position_] == 'E')) {
      ++position_;
      if (position_ < value_.size() && (value_[position_] == '+' || value_[position_] == '-')) {
        ++position_;
      }
      const std::size_t exponent_start = position_;
      while (position_ < value_.size() && value_[position_] >= '0' &&
             value_[position_] <= '9') {
        ++position_;
      }
      if (exponent_start == position_) {
        return false;
      }
    }
    return true;
  }

  bool consume(char character) {
    if (position_ == value_.size() || value_[position_] != character) {
      return false;
    }
    ++position_;
    return true;
  }

  bool consume_literal(const char* literal) {
    for (const char* character = literal; *character != '\0'; ++character) {
      if (!consume(*character)) {
        return false;
      }
    }
    return true;
  }

  const std::string& value_;
  std::size_t position_{0};
};

bool looks_like_json_object(const std::string& value) {
  if (value.size() < 2 || value.front() != '{' || value.back() != '}') {
    return false;
  }
  JsonParser parser(value);
  if (!parser.parse_object_document()) {
    return false;
  }
  bool in_string = false;
  bool escaped = false;
  for (char character : value) {
    const unsigned char byte = static_cast<unsigned char>(character);
    if (byte < 0x20) {
      return false;
    }
    if (in_string && escaped) {
      escaped = false;
    } else if (in_string && character == '\\') {
      escaped = true;
    } else if (character == '"') {
      in_string = !in_string;
    }
  }
  return !in_string && !escaped && value.find("\":") != std::string::npos;
}

bool has_non_rfc4180_record_ending(const std::string& value) {
  bool quoted = false;
  for (std::size_t index = 0; index < value.size(); ++index) {
    if (value[index] == '"') {
      if (quoted && index + 1 < value.size() && value[index + 1] == '"') {
        ++index;
      } else {
        quoted = !quoted;
      }
    } else if (!quoted && value[index] == '\n' &&
               (index == 0 || value[index - 1] != '\r')) {
      return true;
    } else if (!quoted && value[index] == '\r' &&
               (index + 1 == value.size() || value[index + 1] != '\n')) {
      return true;
    }
  }
  return false;
}

std::vector<std::vector<std::string>> csv_records(const std::string& contents) {
  std::vector<std::vector<std::string>> records;
  std::vector<std::string> record;
  std::string field;
  bool quoted = false;
  for (std::size_t index = 0; index < contents.size(); ++index) {
    const char character = contents[index];
    if (quoted) {
      if (character == '"' && index + 1 < contents.size() && contents[index + 1] == '"') {
        field += '"';
        ++index;
      } else if (character == '"') {
        quoted = false;
      } else {
        field += character;
      }
      continue;
    }
    if (character == '"' && field.empty()) {
      quoted = true;
    } else if (character == ',') {
      record.push_back(std::move(field));
      field.clear();
    } else if (character == '\n' || character == '\r') {
      if (character == '\r' && index + 1 < contents.size() && contents[index + 1] == '\n') {
        ++index;
      }
      record.push_back(std::move(field));
      field.clear();
      records.push_back(std::move(record));
      record.clear();
    } else {
      field += character;
    }
  }
  if (!field.empty() || !record.empty()) {
    record.push_back(std::move(field));
    records.push_back(std::move(record));
  }
  return records;
}

void expect_json_event(const std::string& event, const std::string& event_type) {
  EXPECT_TRUE(looks_like_json_object(event));
  EXPECT_TRUE(event.find("\"schema_version\":1") != std::string::npos);
  EXPECT_TRUE(event.find("\"event_type\":\"" + event_type + "\"") !=
              std::string::npos);
  EXPECT_TRUE(event.find(":nan") == std::string::npos);
  EXPECT_TRUE(event.find(":inf") == std::string::npos);
  EXPECT_TRUE(event.find(":-inf") == std::string::npos);
}

rppg_qnn::HeartRateResult make_heart_rate(bool valid) {
  rppg_qnn::HeartRateResult result;
  result.method = "green,method";
  result.window_start_sec = 1.25;
  result.window_end_sec = 11.25;
  result.bpm = 72.5;
  result.confidence = 0.82;
  result.is_valid = valid;
  result.invalid_reason = valid ? "" : "bad,\"quoted\"\nnext";
  result.source_fps = 30.0;
  result.source_frame_count = 300;
  result.max_frame_gap_sec = 0.04;
  result.window_materialization_ms = 1.25;
  result.preprocess_ms = 0.5;
  result.runtime_ms = 1.75;
  result.postprocess_ms = 0.25;
  result.inference_ms = 2.5;
  result.backend = "qnn\"gpu";
  result.model_sha256 = "abc123";
  return result;
}

void test_persisted_events_csv_and_summary() {
  ScopedDirectory directory;
  ScopedCoutCapture output;
  rppg_qnn::ResultSink sink(directory.path());

  rppg_qnn::PreflightResult preflight;
  preflight.qnn_gpu_available = true;
  preflight.opencl_available = true;
  preflight.qnn_gpu_library = "libQnn\"Gpu.so";
  preflight.opencl_library = "libOpenCL.so";
  preflight.error = "quote\" slash\\ backspace\b formfeed\f line\nreturn\r tab\t control\x01";
  sink.publish(preflight);

  sink.publish(make_heart_rate(false));
  rppg_qnn::HeartRateResult non_finite = make_heart_rate(true);
  non_finite.bpm = std::numeric_limits<double>::quiet_NaN();
  non_finite.confidence = std::numeric_limits<double>::infinity();
  non_finite.source_fps = -std::numeric_limits<double>::infinity();
  sink.publish(non_finite);
  sink.close(7);

  const std::vector<std::string> events = json_lines(read_file(directory.path() / "events.jsonl"));
  EXPECT_EQ(events.size(), static_cast<std::size_t>(3));
  if (events.size() == 3) {
    expect_json_event(events[0], "preflight_result");
    expect_json_event(events[1], "heart_rate_result");
    expect_json_event(events[2], "heart_rate_result");
    EXPECT_TRUE(events[0].find("\\\"") != std::string::npos);
    EXPECT_TRUE(events[0].find("\\\\") != std::string::npos);
    EXPECT_TRUE(events[0].find("\\b") != std::string::npos);
    EXPECT_TRUE(events[0].find("\\f") != std::string::npos);
    EXPECT_TRUE(events[0].find("\\n") != std::string::npos);
    EXPECT_TRUE(events[0].find("\\r") != std::string::npos);
    EXPECT_TRUE(events[0].find("\\t") != std::string::npos);
    EXPECT_TRUE(events[0].find("\\u0001") != std::string::npos);
    EXPECT_TRUE(events[1].find("\\\"quoted\\\"") != std::string::npos);
    EXPECT_TRUE(events[1].find("\"window_materialization_ms\":1.25") !=
                std::string::npos);
    EXPECT_TRUE(events[1].find("\"preprocess_ms\":0.5") !=
                std::string::npos);
    EXPECT_TRUE(events[1].find("\"runtime_ms\":1.75") !=
                std::string::npos);
    EXPECT_TRUE(events[1].find("\"postprocess_ms\":0.25") !=
                std::string::npos);
    EXPECT_TRUE(events[2].find("\"bpm\":null") != std::string::npos);
    EXPECT_TRUE(events[2].find("\"confidence\":null") != std::string::npos);
  }

  const std::vector<std::vector<std::string>> records =
      csv_records(read_file(directory.path() / "heart_rate.csv"));
  const std::string csv_contents = read_file(directory.path() / "heart_rate.csv");
  const std::string expected_header =
      "schema_version,method,window_start_sec,window_end_sec,bpm,confidence,is_valid,"
      "invalid_reason,source_fps,source_frame_count,max_frame_gap_sec,"
      "window_materialization_ms,preprocess_ms,runtime_ms,postprocess_ms,inference_ms,"
      "backend,model_sha256";
  EXPECT_EQ(records.size(), static_cast<std::size_t>(3));
  EXPECT_TRUE(csv_contents.find("\r\n") != std::string::npos);
  EXPECT_TRUE(!has_non_rfc4180_record_ending(csv_contents));
  if (records.size() == 3) {
    EXPECT_EQ(records[0].size(), static_cast<std::size_t>(18));
    EXPECT_EQ(records[1].size(), static_cast<std::size_t>(18));
    EXPECT_EQ(records[2].size(), static_cast<std::size_t>(18));
    std::string header;
    for (std::size_t index = 0; index < records[0].size(); ++index) {
      if (index != 0) {
        header += ',';
      }
      header += records[0][index];
    }
    EXPECT_EQ(header, expected_header);
    EXPECT_EQ(records[1][1], std::string("green,method"));
    EXPECT_EQ(records[1][7], std::string("bad,\"quoted\"\nnext"));
    EXPECT_EQ(records[2][4], std::string(""));
    EXPECT_EQ(records[2][5], std::string(""));
    EXPECT_EQ(records[2][8], std::string(""));
  }

  const std::string summary = read_file(directory.path() / "session_summary.json");
  EXPECT_TRUE(looks_like_json_object(summary));
  EXPECT_TRUE(summary.find("\"event_type\":\"session_end\"") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":7") != std::string::npos);
  EXPECT_TRUE(summary.find("\"preflight_count\":1") != std::string::npos);
  EXPECT_TRUE(summary.find("\"frame_health_count\":0") != std::string::npos);
  EXPECT_TRUE(summary.find("\"runtime_error_count\":0") != std::string::npos);
  EXPECT_TRUE(summary.find("\"heart_rate_count\":2") != std::string::npos);
  EXPECT_TRUE(summary.find("\"valid_heart_rate_count\":1") != std::string::npos);
  EXPECT_TRUE(summary.find("\"invalid_heart_rate_count\":1") != std::string::npos);
  EXPECT_TRUE(!std::filesystem::exists(directory.path() / "session_summary.json.tmp"));
  const std::string terminal_output = output.str();
  EXPECT_EQ(static_cast<std::size_t>(std::count(terminal_output.begin(), terminal_output.end(), '\n')),
            static_cast<std::size_t>(2));

  EXPECT_APP_ERROR(sink.publish(preflight), rppg_qnn::ErrorCode::OutputWriteFailed);
  sink.close(7);
}

void test_frame_health_is_batched_on_one_second_boundaries() {
  ScopedDirectory directory;
  ScopedCoutCapture output;
  rppg_qnn::ResultSink sink(directory.path());
  rppg_qnn::FrameHealth frame;
  for (double timestamp : {0.0, 0.99, 1.0, 1.5, 2.0,
                           std::numeric_limits<double>::quiet_NaN(), 1.0}) {
    frame.timestamp_sec = timestamp;
    frame.frame_id++;
    sink.publish(frame);
  }
  sink.close(0);

  const std::vector<std::string> events = json_lines(read_file(directory.path() / "events.jsonl"));
  EXPECT_EQ(events.size(), static_cast<std::size_t>(3));
  if (events.size() == 3) {
    expect_json_event(events[0], "frame_health");
    expect_json_event(events[1], "frame_health");
    expect_json_event(events[2], "frame_health");
    EXPECT_TRUE(events[0].find("\"timestamp_sec\":0") != std::string::npos);
    EXPECT_TRUE(events[1].find("\"timestamp_sec\":1") != std::string::npos);
    EXPECT_TRUE(events[2].find("\"timestamp_sec\":2") != std::string::npos);
  }
  const std::vector<std::vector<std::string>> records =
      csv_records(read_file(directory.path() / "heart_rate.csv"));
  EXPECT_EQ(records.size(), static_cast<std::size_t>(1));
  const std::string summary = read_file(directory.path() / "session_summary.json");
  EXPECT_TRUE(summary.find("\"frame_health_count\":3") != std::string::npos);
  EXPECT_TRUE(summary.find("\"heart_rate_count\":0") != std::string::npos);
  const std::string terminal = output.str();
  EXPECT_EQ(static_cast<std::size_t>(std::count(terminal.begin(), terminal.end(), '\n')),
            static_cast<std::size_t>(3));
  EXPECT_TRUE(terminal.find("frame_health frame_id=1 timestamp_sec=0") !=
              std::string::npos);
  EXPECT_TRUE(terminal.find("status=sampling") != std::string::npos);
}

void test_runtime_errors_are_machine_readable() {
  ScopedDirectory directory;
  rppg_qnn::ResultSink sink(directory.path());
  sink.publish_runtime_error("UNEXPECTED_EXCEPTION", "bad \"message\"\nnext");
  sink.close(1);
  const std::vector<std::string> events = json_lines(read_file(directory.path() / "events.jsonl"));
  const std::string summary = read_file(directory.path() / "session_summary.json");
  EXPECT_EQ(events.size(), static_cast<std::size_t>(1));
  if (events.size() == 1) {
    expect_json_event(events.front(), "runtime_error");
    EXPECT_TRUE(events.front().find("\"error_code\":\"UNEXPECTED_EXCEPTION\"") !=
                std::string::npos);
    EXPECT_TRUE(events.front().find("\\\"message\\\"") != std::string::npos);
  }
  EXPECT_TRUE(summary.find("\"runtime_error_count\":1") != std::string::npos);
}

void test_constructor_failure_and_idempotent_close() {
  ScopedDirectory directory;
  {
    std::ofstream blocking_file(directory.path());
    blocking_file << "not a directory";
  }
  EXPECT_APP_ERROR(rppg_qnn::ResultSink(directory.path()),
                   rppg_qnn::ErrorCode::OutputWriteFailed);

  ScopedDirectory normal_directory;
  rppg_qnn::ResultSink sink(normal_directory.path());
  sink.close(0);
  sink.close(1);
  EXPECT_TRUE(std::filesystem::exists(normal_directory.path() / "session_summary.json"));
}

void test_machine_output_ignores_global_comma_locale() {
  ScopedDirectory directory;
  ScopedGlobalLocale comma_locale(std::locale(std::locale(), new CommaNumpunct));
  ScopedCoutCapture terminal;
  rppg_qnn::ResultSink sink(directory.path());
  rppg_qnn::FrameHealth health;
  health.timestamp_sec = 0.0;
  health.capture_fps = 72.5;
  sink.publish(health);
  rppg_qnn::HeartRateResult result = make_heart_rate(true);
  sink.publish(result);
  sink.close(0);

  const std::string events = read_file(directory.path() / "events.jsonl");
  const std::string csv = read_file(directory.path() / "heart_rate.csv");
  EXPECT_TRUE(events.find("\"bpm\":72.5") != std::string::npos);
  EXPECT_TRUE(events.find("72,5") == std::string::npos);
  EXPECT_TRUE(csv.find("72.5") != std::string::npos);
  EXPECT_EQ(csv_records(csv)[1].size(), static_cast<std::size_t>(18));
  EXPECT_TRUE(terminal.str().find("capture_fps=72.5") != std::string::npos);
}

void test_invalid_utf8_is_replaced_in_json_and_csv() {
  ScopedDirectory directory;
  rppg_qnn::ResultSink sink(directory.path());
  rppg_qnn::HeartRateResult result = make_heart_rate(false);
  std::string invalid_utf8;
  invalid_utf8 += static_cast<char>(0x80);
  invalid_utf8 += static_cast<char>(0xff);
  invalid_utf8 += static_cast<char>(0xc0);
  invalid_utf8 += static_cast<char>(0xaf);
  invalid_utf8 += static_cast<char>(0xe2);
  invalid_utf8 += static_cast<char>(0x82);
  invalid_utf8 += static_cast<char>(0xed);
  invalid_utf8 += static_cast<char>(0xa0);
  invalid_utf8 += static_cast<char>(0x80);
  invalid_utf8 += static_cast<char>(0xf4);
  invalid_utf8 += static_cast<char>(0x90);
  invalid_utf8 += static_cast<char>(0x80);
  invalid_utf8 += static_cast<char>(0x80);
  result.method = std::string("valid-") + u8"心" + invalid_utf8;
  result.invalid_reason = invalid_utf8;
  sink.publish(result);
  sink.close(0);

  const std::string events = read_file(directory.path() / "events.jsonl");
  const std::string csv = read_file(directory.path() / "heart_rate.csv");
  const std::string replacement("\xef\xbf\xbd");
  EXPECT_TRUE(is_valid_utf8(events));
  EXPECT_TRUE(is_valid_utf8(csv));
  EXPECT_TRUE(events.find(u8"心") != std::string::npos);
  EXPECT_TRUE(events.find(replacement) != std::string::npos);
  EXPECT_TRUE(csv.find(replacement) != std::string::npos);
  EXPECT_TRUE(events.find(invalid_utf8) == std::string::npos);
  EXPECT_TRUE(csv.find(invalid_utf8) == std::string::npos);
}

void test_summary_rename_failure_is_retryable_with_first_exit_code() {
  ScopedDirectory directory;
  rppg_qnn::ResultSink sink(directory.path());
  sink.publish(make_heart_rate(true));
  const std::filesystem::path summary_path = directory.path() / "session_summary.json";
  std::filesystem::create_directory(summary_path);
  EXPECT_APP_ERROR(sink.close(23), rppg_qnn::ErrorCode::OutputWriteFailed);
  std::filesystem::remove(summary_path);
  sink.close(99);

  const std::string summary = read_file(summary_path);
  EXPECT_TRUE(summary.find("\"exit_code\":23") != std::string::npos);
  EXPECT_TRUE(!std::filesystem::exists(directory.path() / "session_summary.json.tmp"));
  EXPECT_APP_ERROR(sink.publish(make_heart_rate(true)), rppg_qnn::ErrorCode::OutputWriteFailed);
}

void test_multiple_sinks_write_complete_terminal_lines() {
  ScopedDirectory first_directory;
  ScopedDirectory second_directory;
  ScopedCoutCapture output;
  rppg_qnn::ResultSink first(first_directory.path());
  rppg_qnn::ResultSink second(second_directory.path());
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  const auto publish_many = [&ready, &start](rppg_qnn::ResultSink& sink) {
    ready.fetch_add(1);
    while (!start.load()) {
    }
    for (int index = 0; index < 16; ++index) {
      rppg_qnn::HeartRateResult result = make_heart_rate(true);
      result.bpm = 60.0 + static_cast<double>(index);
      sink.publish(result);
    }
    sink.close(0);
  };
  std::thread first_thread(publish_many, std::ref(first));
  std::thread second_thread(publish_many, std::ref(second));
  while (ready.load() != 2) {
  }
  start.store(true);
  first_thread.join();
  second_thread.join();

  std::istringstream lines(output.str());
  std::size_t line_count = 0;
  for (std::string line; std::getline(lines, line);) {
    ++line_count;
    EXPECT_TRUE(line.find("heart_rate bpm=") == 0);
    EXPECT_TRUE(line.find(" confidence=") != std::string::npos);
    EXPECT_TRUE(line.find(" valid=true") != std::string::npos);
  }
  EXPECT_EQ(line_count, static_cast<std::size_t>(32));
}

}  // namespace

int main() {
  test_persisted_events_csv_and_summary();
  test_frame_health_is_batched_on_one_second_boundaries();
  test_runtime_errors_are_machine_readable();
  test_constructor_failure_and_idempotent_close();
  test_machine_output_ignores_global_comma_locale();
  test_invalid_utf8_is_replaced_in_json_and_csv();
  test_summary_rename_failure_is_retryable_with_first_exit_code();
  test_multiple_sinks_write_complete_terminal_lines();
  return test_support::finish();
}
