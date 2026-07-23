#include "rppg_qnn/pipeline.hpp"

#include "rppg_qnn/deep_runtime.hpp"
#include "rppg_qnn/frame_source.hpp"
#include "rppg_qnn/roi_processor.hpp"
#include "rppg_qnn/result_sink.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/wait.h>

#include <opencv2/videoio.hpp>

#include "test_support.hpp"

#ifndef RPPG_LIVE_PATH
#error "RPPG_LIVE_PATH must point to the CLI executable"
#endif

namespace {

class ScopedDirectory {
 public:
  ScopedDirectory()
      : path_(std::filesystem::temp_directory_path() /
              ("rppg_qnn_pipeline_" + std::to_string(
                  std::chrono::steady_clock::now().time_since_epoch().count()))) {
    std::filesystem::create_directories(path_);
  }
  ~ScopedDirectory() {
    std::error_code error;
    std::filesystem::remove_all(path_, error);
  }
  const std::filesystem::path& path() const { return path_; }
 private:
  std::filesystem::path path_;
};

std::string read_file(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}

std::vector<std::string> json_events(const std::string& contents) {
  std::vector<std::string> events;
  std::istringstream input(contents);
  for (std::string event; std::getline(input, event);) {
    EXPECT_TRUE(event.size() >= 2U && event.front() == '{' && event.back() == '}');
    events.push_back(std::move(event));
  }
  return events;
}

std::optional<double> json_number(const std::string& event, const std::string& key) {
  const std::string prefix = "\"" + key + "\":";
  const std::size_t start = event.find(prefix);
  if (start == std::string::npos) {
    return std::nullopt;
  }
  const std::size_t value_start = start + prefix.size();
  const std::size_t value_end = event.find_first_of(",}", value_start);
  if (value_end == std::string::npos) {
    return std::nullopt;
  }
  try {
    std::size_t parsed = 0;
    const double value = std::stod(event.substr(value_start, value_end - value_start), &parsed);
    return parsed == value_end - value_start ? std::optional<double>(value) : std::nullopt;
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

class ScopedCerrCapture {
 public:
  ScopedCerrCapture() : previous_(std::cerr.rdbuf(stream_.rdbuf())) {}
  ~ScopedCerrCapture() { std::cerr.rdbuf(previous_); }
  std::string str() const { return stream_.str(); }
 private:
  std::ostringstream stream_;
  std::streambuf* previous_;
};

void write_pulse_video(const std::filesystem::path& path) {
  cv::VideoWriter writer(path.string(), cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                         30.0, cv::Size(96, 96));
  EXPECT_TRUE(writer.isOpened());
  for (int frame = 0; frame < 480; ++frame) {
    const double timestamp = static_cast<double>(frame) / 30.0;
    const int green = static_cast<int>(std::lround(
        100.0 + 2.0 * std::sin(2.0 * 3.14159265358979323846 * 1.3 * timestamp)));
    cv::Mat image(96, 96, CV_8UC3, cv::Scalar(25, 25, 25));
    image(cv::Rect(16, 20, 64, 64)).setTo(cv::Scalar(30, green, 20));
    writer.write(image);
  }
}

class FixedRoi final : public rppg_qnn::IRoiProcessor {
 public:
  rppg_qnn::RoiPacket process(const rppg_qnn::FramePacket& frame) override {
    return {frame.frame_id, frame.timestamp_sec, frame.bgr(cv::Rect(16, 20, 64, 64)).clone(),
            rppg_qnn::FaceBox{16, 20, 64, 64, 1.0}, false};
  }
};

class ThrowingRoi final : public rppg_qnn::IRoiProcessor {
 public:
  rppg_qnn::RoiPacket process(const rppg_qnn::FramePacket&) override {
    throw std::runtime_error("deterministic ROI failure");
  }
};

class CountingSource final : public rppg_qnn::FrameSource {
 public:
  CountingSource(std::unique_ptr<rppg_qnn::FrameSource> source, int* reads)
      : source_(std::move(source)), reads_(reads) {}
  std::optional<rppg_qnn::FramePacket> read() override {
    const auto frame = source_->read();
    if (frame.has_value()) {
      ++*reads_;
    }
    return frame;
  }
  bool eof() const override { return source_->eof(); }
  double nominal_fps() const override { return source_->nominal_fps(); }
 private:
  std::unique_ptr<rppg_qnn::FrameSource> source_;
  int* reads_;
};

class SingleFrameSource final : public rppg_qnn::FrameSource {
 public:
  std::optional<rppg_qnn::FramePacket> read() override {
    if (read_) {
      return std::nullopt;
    }
    read_ = true;
    return rppg_qnn::FramePacket{0, 0.0, cv::Mat(96, 96, CV_8UC3, cv::Scalar())};
  }
  bool eof() const override { return read_; }
  double nominal_fps() const override { return 30.0; }
 private:
  bool read_{false};
};

class RecoveringSource final : public rppg_qnn::FrameSource {
 public:
  std::optional<rppg_qnn::FramePacket> read() override {
    if (calls_++ == 0) {
      return std::nullopt;
    }
    if (calls_ == 2) {
      return rppg_qnn::FramePacket{0, 0.0, cv::Mat(96, 96, CV_8UC3, cv::Scalar())};
    }
    eof_ = true;
    return std::nullopt;
  }
  bool eof() const override { return eof_; }
  double nominal_fps() const override { return 30.0; }
 private:
  int calls_{0};
  bool eof_{false};
};

class PermanentEmptySource final : public rppg_qnn::FrameSource {
 public:
  std::optional<rppg_qnn::FramePacket> read() override { return std::nullopt; }
  bool eof() const override { return false; }
  double nominal_fps() const override { return 30.0; }
};

class FastSource final : public rppg_qnn::FrameSource {
 public:
  explicit FastSource(std::chrono::steady_clock::time_point* final_read)
      : final_read_(final_read) {}
  std::optional<rppg_qnn::FramePacket> read() override {
    if (index_ == 120) {
      *final_read_ = std::chrono::steady_clock::now();
      eof_ = true;
      return std::nullopt;
    }
    return rppg_qnn::FramePacket{static_cast<std::uint64_t>(index_),
                                  index_++ / 30.0,
                                  cv::Mat(96, 96, CV_8UC3, cv::Scalar(1, 2, 3))};
  }
  bool eof() const override { return eof_; }
  double nominal_fps() const override { return 30.0; }
 private:
  int index_{0};
  bool eof_{false};
  std::chrono::steady_clock::time_point* final_read_;
};

class SlowSink final : public rppg_qnn::IResultSink {
 public:
  void publish(const rppg_qnn::PreflightResult&) override {}
  void publish_runtime_error(const std::string&, const std::string&) override {}
  void publish(const rppg_qnn::FrameHealth&) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  void publish(const rppg_qnn::HeartRateResult&) override {}
  void close(int) override {}
};

class EmptyThrowSink final : public rppg_qnn::IResultSink {
 public:
  void publish(const rppg_qnn::PreflightResult&) override {}
  void publish_runtime_error(const std::string&, const std::string&) override {}
  void publish(const rppg_qnn::FrameHealth&) override { throw std::runtime_error(""); }
  void publish(const rppg_qnn::HeartRateResult&) override {}
  void close(int) override {}
};

class HeartRateFloodSource final : public rppg_qnn::FrameSource {
 public:
  std::optional<rppg_qnn::FramePacket> read() override {
    if (index_ == 300) {
      eof_ = true;
      return std::nullopt;
    }
    return rppg_qnn::FramePacket{static_cast<std::uint64_t>(index_),
                                  static_cast<double>(index_++),
                                  cv::Mat(96, 96, CV_8UC3, cv::Scalar(1, 100, 3))};
  }
  bool eof() const override { return eof_; }
  double nominal_fps() const override { return 30.0; }
 private:
  int index_{0};
  bool eof_{false};
};

class DeepScheduleSource final : public rppg_qnn::FrameSource {
 public:
  std::optional<rppg_qnn::FramePacket> read() override {
    if (index_ == 211) {
      eof_ = true;
      return std::nullopt;
    }
    return rppg_qnn::FramePacket{static_cast<std::uint64_t>(index_),
                                  index_++ / 30.0,
                                  cv::Mat(96, 96, CV_8UC3, cv::Scalar(1, 100, 3))};
  }
  bool eof() const override { return eof_; }
  double nominal_fps() const override { return 30.0; }
 private:
  int index_{0};
  bool eof_{false};
};

class TraditionalPulseSource final : public rppg_qnn::FrameSource {
 public:
  std::optional<rppg_qnn::FramePacket> read() override {
    if (index_ == 480) {
      eof_ = true;
      return std::nullopt;
    }
    const double time = static_cast<double>(index_) / 30.0;
    const double pulse = std::sin(2.0 * 3.14159265358979323846 * 1.2 * time);
    const double motion = std::sin(2.0 * 3.14159265358979323846 * 0.2 * time);
    const double red = 120.0 * (1.0 + 0.04 * motion + 0.002 * pulse);
    const double green = 90.0 * (1.0 + 0.04 * motion + 0.010 * pulse);
    const double blue = 70.0 * (1.0 + 0.04 * motion - 0.003 * pulse);
    cv::Mat frame(96, 96, CV_8UC3, cv::Scalar(25, 25, 25));
    frame(cv::Rect(16, 20, 64, 64))
        .setTo(cv::Scalar(std::round(blue), std::round(green), std::round(red)));
    return rppg_qnn::FramePacket{static_cast<std::uint64_t>(index_),
                                  static_cast<double>(index_++) / 30.0,
                                  std::move(frame)};
  }
  bool eof() const override { return eof_; }
  double nominal_fps() const override { return 30.0; }

 private:
  int index_{0};
  bool eof_{false};
};

class InvalidRoiAtSevenSeconds final : public rppg_qnn::IRoiProcessor {
 public:
  rppg_qnn::RoiPacket process(const rppg_qnn::FramePacket& frame) override {
    cv::Mat roi(64, 64, frame.frame_id == 210U ? CV_8UC1 : CV_8UC3,
                cv::Scalar(1, 100, 3));
    return {frame.frame_id, frame.timestamp_sec, std::move(roi),
            rppg_qnn::FaceBox{16, 20, 64, 64, 1.0}, false};
  }
};

class RecordingDeepRuntime final : public rppg_qnn::IDeepRuntime {
 public:
  explicit RecordingDeepRuntime(std::vector<double>* input_ends) : input_ends_(input_ends) {}
  [[nodiscard]] std::string backend_name() const override { return "recording"; }
  rppg_qnn::HeartRateResult infer(const rppg_qnn::DeepInput& input) override {
    input_ends_->push_back(input.end_sec);
    rppg_qnn::HeartRateResult result;
    result.method = "RECORDING";
    result.backend = backend_name();
    result.window_start_sec = input.start_sec;
    result.window_end_sec = input.end_sec;
    result.source_fps = input.source_fps;
    result.source_frame_count = input.source_frame_count;
    result.max_frame_gap_sec = input.max_frame_gap_sec;
    return result;
  }
 private:
  std::vector<double>* input_ends_;
};

class SlowResultSink final : public rppg_qnn::IResultSink {
 public:
  explicit SlowResultSink(const std::filesystem::path& path) : inner_(path) {}
  void publish(const rppg_qnn::PreflightResult& result) override { inner_.publish(result); }
  void publish_runtime_error(const std::string& code, const std::string& message) override {
    inner_.publish_runtime_error(code, message);
  }
  void publish(const rppg_qnn::FrameHealth& result) override { inner_.publish(result); }
  void publish(const rppg_qnn::HeartRateResult& result) override {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    inner_.publish(result);
  }
  void close(int exit_code) override { inner_.close(exit_code); }
 private:
  rppg_qnn::ResultSink inner_;
};

rppg_qnn::PipelineDependencies video_dependencies(const std::filesystem::path& video,
                                                   int* reads,
                                                   bool* deep_factory_called) {
  return {
      [video, reads] {
        return std::make_unique<CountingSource>(rppg_qnn::make_video_source(video), reads);
      },
      [] { return std::make_unique<FixedRoi>(); },
      [deep_factory_called] {
        *deep_factory_called = true;
        return rppg_qnn::make_fake_deep_runtime(std::chrono::milliseconds(0));
      },
      {}};
}

void synthetic_video_runs_green_and_fake_deep_to_completion() {
  ScopedDirectory directory;
  const std::filesystem::path video = directory.path() / "pulse.avi";
  write_pulse_video(video);
  int reads = 0;
  bool deep_factory_called = false;
  rppg_qnn::AppConfig config;
  config.video = video.string();
  config.deep = "fake";
  config.output = directory.path() / "deep";
  rppg_qnn::Pipeline pipeline(config, video_dependencies(video, &reads, &deep_factory_called));

  EXPECT_EQ(pipeline.run(), 0);
  EXPECT_TRUE(deep_factory_called);
  EXPECT_EQ(reads, 480);
  const std::string events = read_file(config.output / "events.jsonl");
  const std::string csv = read_file(config.output / "heart_rate.csv");
  const std::string summary = read_file(config.output / "session_summary.json");
  EXPECT_TRUE(!events.empty());
  EXPECT_TRUE(!csv.empty());
  bool valid_green = false;
  bool fake_deep = false;
  for (const std::string& event : json_events(events)) {
    if (event.find("\"method\":\"FAKE_DEEP\"") != std::string::npos) {
      fake_deep = true;
    }
    if (event.find("\"method\":\"GREEN\"") != std::string::npos &&
        event.find("\"is_valid\":true") != std::string::npos) {
      const std::optional<double> bpm = json_number(event, "bpm");
      valid_green = bpm.has_value() && *bpm >= 75.0 && *bpm <= 81.0;
    }
  }
  EXPECT_TRUE(fake_deep);
  EXPECT_TRUE(valid_green);
  EXPECT_TRUE(summary.find("\"exit_code\":0") != std::string::npos);
}

void configured_pos_and_chrom_are_emitted_without_green_fallback() {
  ScopedDirectory directory;
  for (const std::string& method : {std::string("pos"), std::string("chrom")}) {
    rppg_qnn::AppConfig config;
    config.video = "synthetic.avi";
    config.traditional = method;
    config.deep = "disabled";
    config.output = directory.path() / method;
    rppg_qnn::Pipeline pipeline(
        config,
        {[] { return std::make_unique<TraditionalPulseSource>(); },
         [] { return std::make_unique<FixedRoi>(); }, {}, {}});

    EXPECT_EQ(pipeline.run(), 0);
    const std::string events = read_file(config.output / "events.jsonl");
    const std::string expected_method = method == "pos" ? "POS" : "CHROM";
    EXPECT_TRUE(events.find("\"method\":\"" + expected_method + "\"") !=
                std::string::npos);
    EXPECT_TRUE(events.find("\"method\":\"GREEN\"") == std::string::npos);
    bool valid_72_bpm = false;
    for (const std::string& event : json_events(events)) {
      if (event.find("\"method\":\"" + expected_method + "\"") !=
              std::string::npos &&
          event.find("\"is_valid\":true") != std::string::npos) {
        const std::optional<double> bpm = json_number(event, "bpm");
        valid_72_bpm = bpm.has_value() && std::abs(*bpm - 72.0) <= 0.01;
      }
    }
    EXPECT_TRUE(valid_72_bpm);
  }
}

void disabled_deep_never_constructs_runtime_or_loses_frames() {
  ScopedDirectory directory;
  const std::filesystem::path video = directory.path() / "pulse.avi";
  write_pulse_video(video);
  int reads = 0;
  bool deep_factory_called = false;
  rppg_qnn::AppConfig config;
  config.video = video.string();
  config.deep = "disabled";
  config.output = directory.path() / "disabled";
  rppg_qnn::PipelineDependencies dependencies{
      [video, &reads] {
        return std::make_unique<CountingSource>(rppg_qnn::make_video_source(video), &reads);
      },
      [] { return std::make_unique<FixedRoi>(); },
      {},
      {}};
  EXPECT_TRUE(!dependencies.make_deep_runtime);
  rppg_qnn::Pipeline pipeline(config, std::move(dependencies));

  EXPECT_EQ(pipeline.run(), 0);
  EXPECT_TRUE(!deep_factory_called);
  EXPECT_EQ(reads, 480);
  EXPECT_TRUE(read_file(config.output / "events.jsonl").find("FAKE_DEEP") == std::string::npos);
}

void preflight_does_not_construct_capture_or_roi() {
  ScopedDirectory directory;
  rppg_qnn::AppConfig config;
  config.preflight_only = true;
  config.qnn_gpu_library = "/definitely/missing/libQnnGpu.so";
  config.opencl_library = "/definitely/missing/libOpenCL.so";
  config.output = directory.path() / "preflight";
  rppg_qnn::PipelineDependencies dependencies{};
  rppg_qnn::Pipeline pipeline(config, std::move(dependencies));

  ScopedCerrCapture stderr_capture;
  EXPECT_EQ(pipeline.run(), 7);
  const std::string events = read_file(config.output / "events.jsonl");
  const std::string summary = read_file(config.output / "session_summary.json");
  EXPECT_TRUE(events.find("\"event_type\":\"preflight_result\"") != std::string::npos);
  EXPECT_TRUE(events.find("QNN_LIBRARY_NOT_FOUND") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":7") != std::string::npos);
  const std::string error = stderr_capture.str();
  EXPECT_TRUE(error.find("QNN_LIBRARY_NOT_FOUND:") == 0U);
  EXPECT_EQ(static_cast<std::size_t>(std::count(error.begin(), error.end(), '\n')),
            static_cast<std::size_t>(1));
}

void runtime_failures_are_reported_and_closed() {
  ScopedDirectory directory;
  rppg_qnn::AppConfig config;
  config.video = "unused.avi";
  config.output = directory.path() / "error";
  rppg_qnn::PipelineDependencies dependencies{
      []() -> std::unique_ptr<rppg_qnn::FrameSource> {
        throw std::runtime_error("deterministic source failure");
      },
      [] { return std::make_unique<ThrowingRoi>(); },
      [] { return rppg_qnn::make_fake_deep_runtime(std::chrono::milliseconds(0)); },
      {}};
  rppg_qnn::Pipeline pipeline(config, std::move(dependencies));

  ScopedCerrCapture stderr_capture;
  EXPECT_TRUE(pipeline.run() != 0);
  const std::string events = read_file(config.output / "events.jsonl");
  const std::string summary = read_file(config.output / "session_summary.json");
  EXPECT_TRUE(events.find("\"event_type\":\"runtime_error\"") != std::string::npos);
  EXPECT_TRUE(events.find("\"error_code\":\"UNEXPECTED_EXCEPTION\"") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":0") == std::string::npos);
  const std::string error = stderr_capture.str();
  EXPECT_TRUE(error.find("UNEXPECTED_EXCEPTION: deterministic source failure\n") !=
              std::string::npos);
  EXPECT_EQ(static_cast<std::size_t>(std::count(error.begin(), error.end(), '\n')),
            static_cast<std::size_t>(1));
}

void transient_empty_reads_recover_but_permanent_empty_reads_fail() {
  ScopedDirectory directory;
  rppg_qnn::AppConfig recovered_config;
  recovered_config.video = "unused.avi";
  recovered_config.output = directory.path() / "recovered";
  rppg_qnn::Pipeline recovered(
      recovered_config,
      {[] { return std::make_unique<RecoveringSource>(); },
       [] { return std::make_unique<FixedRoi>(); }, {}, {}});
  EXPECT_EQ(recovered.run(), 0);

  rppg_qnn::AppConfig failed_config;
  failed_config.video = "unused.avi";
  failed_config.output = directory.path() / "empty";
  rppg_qnn::Pipeline failed(
      failed_config,
      {[] { return std::make_unique<PermanentEmptySource>(); },
       [] { return std::make_unique<FixedRoi>(); }, {}, {}});
  ScopedCerrCapture stderr_capture;
  EXPECT_EQ(failed.run(), 3);
  EXPECT_TRUE(read_file(failed_config.output / "events.jsonl").find("runtime_error") !=
              std::string::npos);
  EXPECT_TRUE(read_file(failed_config.output / "session_summary.json").find("\"exit_code\":3") !=
              std::string::npos);
}

void slow_output_does_not_hold_up_capture() {
  ScopedDirectory directory;
  std::chrono::steady_clock::time_point final_read{};
  rppg_qnn::AppConfig config;
  config.video = "unused.avi";
  config.output = directory.path() / "slow";
  rppg_qnn::Pipeline pipeline(
      config,
      {[&final_read] { return std::make_unique<FastSource>(&final_read); },
       [] { return std::make_unique<FixedRoi>(); }, {},
       [](const std::filesystem::path&) { return std::make_unique<SlowSink>(); }});
  EXPECT_EQ(pipeline.run(), 0);
  EXPECT_TRUE(std::chrono::steady_clock::now() - final_read >= std::chrono::milliseconds(80));
}

void empty_output_exception_never_becomes_a_successful_session() {
  ScopedDirectory directory;
  rppg_qnn::AppConfig config;
  config.video = "unused.avi";
  config.output = directory.path() / "empty-throw";
  rppg_qnn::Pipeline pipeline(
      config,
      {[] { return std::make_unique<SingleFrameSource>(); },
       [] { return std::make_unique<FixedRoi>(); }, {},
       [](const std::filesystem::path&) { return std::make_unique<EmptyThrowSink>(); }});
  ScopedCerrCapture stderr_capture;
  EXPECT_EQ(pipeline.run(), 13);
  EXPECT_TRUE(stderr_capture.str().find("OUTPUT_WRITE_FAILED:") == 0U);
}

void a_full_heart_rate_queue_reserves_one_runtime_error_slot() {
  ScopedDirectory directory;
  rppg_qnn::AppConfig config;
  config.video = "unused.avi";
  config.output = directory.path() / "queue-full";
  rppg_qnn::Pipeline pipeline(
      config,
      {[] { return std::make_unique<HeartRateFloodSource>(); },
       [] { return std::make_unique<FixedRoi>(); }, {},
       [](const std::filesystem::path& path) { return std::make_unique<SlowResultSink>(path); }});
  ScopedCerrCapture stderr_capture;
  EXPECT_EQ(pipeline.run(), 13);
  const std::string summary = read_file(config.output / "session_summary.json");
  EXPECT_TRUE(summary.find("\"exit_code\":13") != std::string::npos);
  EXPECT_TRUE(summary.find("\"runtime_error_count\":1") != std::string::npos);
}

void rejected_roi_does_not_schedule_a_stale_deep_window() {
  ScopedDirectory directory;
  std::vector<double> input_ends;
  rppg_qnn::AppConfig config;
  config.video = "unused.avi";
  config.deep = "fake";
  config.output = directory.path() / "invalid-roi";
  rppg_qnn::Pipeline pipeline(
      config,
      {[] { return std::make_unique<DeepScheduleSource>(); },
       [] { return std::make_unique<InvalidRoiAtSevenSeconds>(); },
       [&input_ends] { return std::make_unique<RecordingDeepRuntime>(&input_ends); }, {}});

  EXPECT_EQ(pipeline.run(), 0);
  EXPECT_EQ(input_ends.size(), static_cast<std::size_t>(1));
  if (!input_ends.empty()) {
    EXPECT_EQ(input_ends.front(), 6.0);
  }
}

void slow_fake_runtime_keeps_every_capture_frame() {
  ScopedDirectory directory;
  const std::filesystem::path video = directory.path() / "pulse.avi";
  write_pulse_video(video);
  int reads = 0;
  rppg_qnn::AppConfig config;
  config.video = video.string();
  config.deep = "fake";
  config.output = directory.path() / "slow-deep";
  rppg_qnn::Pipeline pipeline(
      config,
      {[video, &reads] {
         return std::make_unique<CountingSource>(rppg_qnn::make_video_source(video), &reads);
       },
       [] { return std::make_unique<FixedRoi>(); },
       [] { return rppg_qnn::make_fake_deep_runtime(std::chrono::milliseconds(100)); },
       {}});
  const auto started = std::chrono::steady_clock::now();
  EXPECT_EQ(pipeline.run(), 0);
  EXPECT_EQ(reads, 480);
  EXPECT_TRUE(std::chrono::steady_clock::now() - started < std::chrono::seconds(5));
  EXPECT_TRUE(read_file(config.output / "events.jsonl").find("FAKE_DEEP") !=
              std::string::npos);
}

void roi_runtime_failures_are_reported_and_closed() {
  ScopedDirectory directory;
  rppg_qnn::AppConfig config;
  config.video = "unused.avi";
  config.deep = "fake";
  config.output = directory.path() / "roi-error";
  rppg_qnn::PipelineDependencies dependencies{
      [] { return std::make_unique<SingleFrameSource>(); },
      [] { return std::make_unique<ThrowingRoi>(); },
      [] { return rppg_qnn::make_fake_deep_runtime(std::chrono::milliseconds(0)); },
      {}};
  rppg_qnn::Pipeline pipeline(config, std::move(dependencies));

  EXPECT_TRUE(pipeline.run() != 0);
  const std::string events = read_file(config.output / "events.jsonl");
  const std::string summary = read_file(config.output / "session_summary.json");
  EXPECT_TRUE(events.find("\"event_type\":\"runtime_error\"") != std::string::npos);
  EXPECT_TRUE(events.find("deterministic ROI failure") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":0") == std::string::npos);
}

void cli_preflight_only_reports_unavailable_qnn_without_a_camera_or_cascade() {
  ScopedDirectory directory;
  const std::filesystem::path output = directory.path() / "cli";
  const std::filesystem::path stderr_path = directory.path() / "stderr.txt";
  const std::string command = std::string("\"") + RPPG_LIVE_PATH +
                              "\" --preflight-only --qnn-gpu-library /definitely/missing/libQnnGpu.so"
                              " --opencl-library /definitely/missing/libOpenCL.so --output \"" +
                              output.string() + "\" 2> \"" + stderr_path.string() + "\"";
  const int status = std::system(command.c_str());
  EXPECT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 7);
  const std::string events = read_file(output / "events.jsonl");
  const std::string summary = read_file(output / "session_summary.json");
  const std::string error = read_file(stderr_path);
  EXPECT_TRUE(events.find("\"event_type\":\"preflight_result\"") != std::string::npos);
  EXPECT_TRUE(events.find("QNN_LIBRARY_NOT_FOUND") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":7") != std::string::npos);
  EXPECT_TRUE(error.find("QNN_LIBRARY_NOT_FOUND:") == 0U);
  EXPECT_EQ(static_cast<std::size_t>(std::count(error.begin(), error.end(), '\n')),
            static_cast<std::size_t>(1));
}

}  // namespace

int main() {
  synthetic_video_runs_green_and_fake_deep_to_completion();
  configured_pos_and_chrom_are_emitted_without_green_fallback();
  disabled_deep_never_constructs_runtime_or_loses_frames();
  preflight_does_not_construct_capture_or_roi();
  runtime_failures_are_reported_and_closed();
  transient_empty_reads_recover_but_permanent_empty_reads_fail();
  slow_output_does_not_hold_up_capture();
  empty_output_exception_never_becomes_a_successful_session();
  a_full_heart_rate_queue_reserves_one_runtime_error_slot();
  rejected_roi_does_not_schedule_a_stale_deep_window();
  slow_fake_runtime_keeps_every_capture_frame();
  roi_runtime_failures_are_reported_and_closed();
  cli_preflight_only_reports_unavailable_qnn_without_a_camera_or_cascade();
  return test_support::finish();
}
