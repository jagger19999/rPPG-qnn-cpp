#include "rppg_qnn/pipeline.hpp"

#include "rppg_qnn/deep_runtime.hpp"
#include "rppg_qnn/frame_source.hpp"
#include "rppg_qnn/roi_processor.hpp"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

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
      }};
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
  EXPECT_TRUE(events.find("\"method\":\"FAKE_DEEP\"") != std::string::npos);
  EXPECT_TRUE(events.find("\"method\":\"GREEN\"") != std::string::npos);
  EXPECT_TRUE(events.find("\"bpm\":78") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":0") != std::string::npos);
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
  rppg_qnn::Pipeline pipeline(config, video_dependencies(video, &reads, &deep_factory_called));

  EXPECT_EQ(pipeline.run(), 0);
  EXPECT_TRUE(!deep_factory_called);
  EXPECT_EQ(reads, 480);
  EXPECT_TRUE(read_file(config.output / "events.jsonl").find("FAKE_DEEP") == std::string::npos);
}

void preflight_does_not_construct_capture_or_roi() {
  ScopedDirectory directory;
  bool source_called = false;
  bool roi_called = false;
  rppg_qnn::AppConfig config;
  config.preflight_only = true;
  config.output = directory.path() / "preflight";
  rppg_qnn::PipelineDependencies dependencies{
      [&source_called]() -> std::unique_ptr<rppg_qnn::FrameSource> {
        source_called = true;
        return {};
      },
      [&roi_called]() -> std::unique_ptr<rppg_qnn::IRoiProcessor> {
        roi_called = true;
        return {};
      },
      []() -> std::unique_ptr<rppg_qnn::IDeepRuntime> { return {}; }};
  rppg_qnn::Pipeline pipeline(config, std::move(dependencies));

  EXPECT_EQ(pipeline.run(), 0);
  EXPECT_TRUE(!source_called);
  EXPECT_TRUE(!roi_called);
  EXPECT_TRUE(!read_file(config.output / "session_summary.json").empty());
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
      [] { return rppg_qnn::make_fake_deep_runtime(std::chrono::milliseconds(0)); }};
  rppg_qnn::Pipeline pipeline(config, std::move(dependencies));

  EXPECT_TRUE(pipeline.run() != 0);
  const std::string events = read_file(config.output / "events.jsonl");
  const std::string summary = read_file(config.output / "session_summary.json");
  EXPECT_TRUE(events.find("\"event_type\":\"runtime_error\"") != std::string::npos);
  EXPECT_TRUE(events.find("\"error_code\":\"UNEXPECTED_EXCEPTION\"") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":0") == std::string::npos);
}

void roi_runtime_failures_are_reported_and_closed() {
  ScopedDirectory directory;
  rppg_qnn::AppConfig config;
  config.video = "unused.avi";
  config.output = directory.path() / "roi-error";
  rppg_qnn::PipelineDependencies dependencies{
      [] { return std::make_unique<SingleFrameSource>(); },
      [] { return std::make_unique<ThrowingRoi>(); },
      [] { return rppg_qnn::make_fake_deep_runtime(std::chrono::milliseconds(0)); }};
  rppg_qnn::Pipeline pipeline(config, std::move(dependencies));

  EXPECT_TRUE(pipeline.run() != 0);
  const std::string events = read_file(config.output / "events.jsonl");
  const std::string summary = read_file(config.output / "session_summary.json");
  EXPECT_TRUE(events.find("\"event_type\":\"runtime_error\"") != std::string::npos);
  EXPECT_TRUE(events.find("deterministic ROI failure") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":0") == std::string::npos);
}

void cli_preflight_only_writes_a_report_without_a_camera_or_cascade() {
  ScopedDirectory directory;
  const std::filesystem::path output = directory.path() / "cli";
  const std::string command = std::string("\"") + RPPG_LIVE_PATH +
                              "\" --preflight-only --output \"" + output.string() + "\"";
  EXPECT_EQ(std::system(command.c_str()), 0);
  const std::string events = read_file(output / "events.jsonl");
  const std::string summary = read_file(output / "session_summary.json");
  EXPECT_TRUE(events.find("\"event_type\":\"preflight_result\"") != std::string::npos);
  EXPECT_TRUE(summary.find("\"exit_code\":0") != std::string::npos);
}

}  // namespace

int main() {
  synthetic_video_runs_green_and_fake_deep_to_completion();
  disabled_deep_never_constructs_runtime_or_loses_frames();
  preflight_does_not_construct_capture_or_roi();
  runtime_failures_are_reported_and_closed();
  roi_runtime_failures_are_reported_and_closed();
  cli_preflight_only_writes_a_report_without_a_camera_or_cascade();
  return test_support::finish();
}
