# C++ Runtime Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an independent Linux-oriented C++ executable that reads V4L2 cameras or video files, produces a traditional GREEN rPPG result, runs a decoupled deterministic test deep worker, writes JSONL/CSV, and proves whether the target QAIRT/QNN GPU runtime is available.

**Architecture:** A `FrameSource` feeds timestamped frames into one ROI processor. Traditional rPPG and deep-window work consume copied immutable packets through bounded latest-only queues, so neither inference nor output can block capture. Hardware-specific QNN execution remains behind `IDeepRuntime`; this phase ships a deterministic fake runtime plus a real QAIRT library preflight, establishing a compileable boundary for the next EfficientPhys/QNN plan.

**Tech Stack:** C++17, CMake 3.20+, OpenCV C++ (`core`, `imgproc`, `videoio`, `objdetect`), POSIX `dlopen`, Linux V4L2 through OpenCV, CTest, standard library threading/filesystem.

---

## Scope boundary

This is plan 1 of 3:

1. **Runtime foundation (this plan):** capture, ROI, GREEN, queues, outputs, QAIRT preflight, fake deep runtime, packaging skeleton.
2. **EfficientPhys QNN GPU:** frozen export inputs, PyTorch→ONNX→QAIRT conversion, native QNN context loading and numerical alignment.
3. **Bench release:** real QNN integration in the live pipeline, cross-build/toolchain hardening, 30-minute soak test and deploy/rollback package.

The repository `/Users/wangjie/Documents/keti/rPPG` is never modified. All commands run in `/Users/wangjie/Documents/keti/rPPG-qnn-cpp`.

## Planned file map

```text
CMakeLists.txt                              build entry, feature switches and tests
cmake/Toolchains/aarch64-linux.cmake        configurable AArch64 cross toolchain
.gitignore                                  build, model and output exclusions
include/rppg_qnn/contracts.hpp              immutable cross-module data contracts
include/rppg_qnn/error.hpp                  stable error codes and exception type
include/rppg_qnn/config.hpp                 CLI/runtime configuration
include/rppg_qnn/latest_queue.hpp           bounded latest-only handoff
include/rppg_qnn/frame_source.hpp           camera/video input interface
include/rppg_qnn/roi_processor.hpp          face/ROI interface
include/rppg_qnn/green_predictor.hpp        traditional GREEN estimator
include/rppg_qnn/deep_runtime.hpp            replaceable deep inference contract
include/rppg_qnn/deep_window_builder.hpp     timestamp-aware deep input windows
include/rppg_qnn/deep_worker.hpp             asynchronous deep-window worker
include/rppg_qnn/qnn_preflight.hpp           QAIRT/QNN/OpenCL runtime probe
include/rppg_qnn/result_sink.hpp             JSONL/CSV output interface
include/rppg_qnn/pipeline.hpp                lifecycle and thread orchestration
src/config.cpp                               strict argument parser
src/contracts.cpp                            contract translation unit
src/error.cpp                                stable error strings
src/frame_source.cpp                         OpenCV V4L2/video implementation
src/roi_processor.cpp                        face detection and cheek ROI
src/green_predictor.cpp                      resampling, filtering, FFT and BPM
src/deep_runtime.cpp                         deterministic fake deep runtime
src/deep_window_builder.cpp                  30 FPS RGB tensor construction
src/deep_worker.cpp                          latest-only deep scheduling
src/qnn_preflight.cpp                        dlopen/symbol/version preflight
src/result_sink.cpp                          terminal/JSONL/CSV serialization
src/pipeline.cpp                              integrated capture loop
src/main.cpp                                  CLI entry and exit-code mapping
tests/test_support.hpp                        dependency-free test assertions
tests/test_contracts.cpp                      contract and error tests
tests/test_config.cpp                         CLI parser tests
tests/test_latest_queue.cpp                   concurrency/overwrite tests
tests/test_frame_source.cpp                   synthetic video input tests
tests/test_roi_processor.cpp                  deterministic ROI tests
tests/test_green_predictor.cpp                synthetic pulse tests
tests/test_deep_worker.cpp                     non-blocking worker tests
tests/test_deep_window_builder.cpp             deep resampling and layout tests
tests/test_qnn_preflight.cpp                   library probe tests
tests/test_result_sink.cpp                     JSONL/CSV schema tests
tests/test_pipeline.cpp                        end-to-end video smoke test
scripts/build_linux.sh                         reproducible native build
scripts/run_smoke.sh                           generated-video smoke run
packaging/run_rppg_qnn.sh                      controlled runtime library path
README.md                                      build, run, preflight and outputs
```

### Task 1: Create the buildable C++ project skeleton

**Files:**
- Create: `CMakeLists.txt`
- Create: `.gitignore`
- Create: `cmake/Toolchains/aarch64-linux.cmake`
- Create: `src/main.cpp`
- Create: `tests/test_support.hpp`
- Create: `tests/test_contracts.cpp`

- [ ] **Step 1: Write the first failing configure check**

Create `tests/test_contracts.cpp`:

```cpp
#include "test_support.hpp"
#include "rppg_qnn/contracts.hpp"

int main() {
  rppg_qnn::HeartRateResult result{};
  EXPECT_EQ(result.schema_version, 1);
  EXPECT_TRUE(result.method.empty());
  return test_support::finish();
}
```

Create `tests/test_support.hpp`:

```cpp
#pragma once
#include <iostream>

namespace test_support {
inline int failures = 0;
inline void fail(const char* expression, const char* file, int line) {
  ++failures;
  std::cerr << file << ':' << line << " expectation failed: " << expression << '\n';
}
inline int finish() { return failures == 0 ? 0 : 1; }
}

#define EXPECT_TRUE(value) do { if (!(value)) test_support::fail(#value, __FILE__, __LINE__); } while (false)
#define EXPECT_EQ(left, right) do { if (!((left) == (right))) test_support::fail(#left " == " #right, __FILE__, __LINE__); } while (false)
```

- [ ] **Step 2: Add the initial CMake build and verify the missing contract fails**

Create `CMakeLists.txt` with C++17, OpenCV discovery, a `rppg_qnn_core` library, `rppg_qnn_live`, CTest, warnings, and optional sanitizers:

```cmake
cmake_minimum_required(VERSION 3.20)
project(rppg_qnn_cpp VERSION 0.1.0 LANGUAGES CXX)

option(RPPG_ENABLE_SANITIZERS "Enable ASan and UBSan" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio objdetect)

add_library(rppg_qnn_core src/contracts.cpp)
target_include_directories(rppg_qnn_core PUBLIC include)
target_link_libraries(rppg_qnn_core PUBLIC ${OpenCV_LIBS} ${CMAKE_DL_LIBS})
target_compile_options(rppg_qnn_core PRIVATE -Wall -Wextra -Wpedantic -Werror)

if(RPPG_ENABLE_SANITIZERS AND CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
  target_compile_options(rppg_qnn_core PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(rppg_qnn_core PUBLIC -fsanitize=address,undefined)
endif()

add_executable(rppg_qnn_live src/main.cpp)
target_link_libraries(rppg_qnn_live PRIVATE rppg_qnn_core)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_contracts tests/test_contracts.cpp)
  target_include_directories(test_contracts PRIVATE tests)
  target_link_libraries(test_contracts PRIVATE rppg_qnn_core)
  add_test(NAME contracts COMMAND test_contracts)
endif()
```

Run:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build
```

Expected: compilation fails because `rppg_qnn/contracts.hpp` and `src/contracts.cpp` do not exist.

- [ ] **Step 3: Add minimal contracts and executable**

Create `include/rppg_qnn/contracts.hpp`:

```cpp
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace rppg_qnn {
struct HeartRateResult {
  int schema_version{1};
  std::string method;
  double window_start_sec{0.0};
  double window_end_sec{0.0};
  double bpm{0.0};
  double confidence{0.0};
  bool is_valid{false};
  std::string invalid_reason;
  double source_fps{0.0};
  std::size_t source_frame_count{0};
  double max_frame_gap_sec{0.0};
  double inference_ms{0.0};
  std::string backend;
  std::string model_sha256;
  std::vector<float> waveform;
};
}
```

Create `src/contracts.cpp` containing only `#include "rppg_qnn/contracts.hpp"`, and create `src/main.cpp`:

```cpp
#include <iostream>
int main() {
  std::cout << "rppg_qnn_live 0.1.0\n";
  return 0;
}
```

Create `.gitignore`:

```gitignore
/build*/
/models/
/outputs/
/.cache/
*.onnx
*.dlc
*.bin
*.so
```

Create `cmake/Toolchains/aarch64-linux.cmake`:

```cmake
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER "$ENV{AARCH64_TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "$ENV{AARCH64_TOOLCHAIN_PREFIX}g++")
set(CMAKE_FIND_ROOT_PATH "$ENV{AARCH64_SYSROOT}")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
```

- [ ] **Step 4: Build and run the first test**

Run:

```bash
cmake -S . -B build -DBUILD_TESTING=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./build/rppg_qnn_live
```

Expected: one passing test and output `rppg_qnn_live 0.1.0`.

- [ ] **Step 5: Commit**

```bash
git add CMakeLists.txt .gitignore cmake include src tests
git commit -m "chore: initialize independent C++ runtime"
```

### Task 2: Define stable contracts and error codes

**Files:**
- Modify: `include/rppg_qnn/contracts.hpp`
- Create: `include/rppg_qnn/error.hpp`
- Create: `src/error.cpp`
- Modify: `CMakeLists.txt`
- Modify: `tests/test_contracts.cpp`

- [ ] **Step 1: Add failing contract validation tests**

Extend `tests/test_contracts.cpp`:

```cpp
#include "rppg_qnn/error.hpp"

rppg_qnn::FrameHealth health{};
EXPECT_EQ(health.schema_version, 1);
EXPECT_EQ(rppg_qnn::to_string(rppg_qnn::ErrorCode::QnnGpuInitFailed), "QNN_GPU_INIT_FAILED");
rppg_qnn::AppError error(rppg_qnn::ErrorCode::ConfigInvalid, "camera and video conflict");
EXPECT_EQ(error.code(), rppg_qnn::ErrorCode::ConfigInvalid);
EXPECT_TRUE(std::string(error.what()).find("camera and video conflict") != std::string::npos);
```

Run `cmake --build build && ctest --test-dir build --output-on-failure`.

Expected: compilation fails because `FrameHealth`, `ErrorCode`, and `AppError` are undefined.

- [ ] **Step 2: Implement the contracts**

Add to `contracts.hpp`:

```cpp
struct FrameHealth {
  int schema_version{1};
  std::uint64_t frame_id{0};
  double timestamp_sec{0.0};
  double capture_fps{0.0};
  bool face_found{false};
  double face_confidence{0.0};
  std::string status{"sampling"};
};

struct PreflightResult {
  int schema_version{1};
  bool qnn_gpu_available{false};
  bool opencl_available{false};
  std::string qnn_gpu_library;
  std::string opencl_library;
  std::string error;
};
```

Create `error.hpp` with this stable enum and an `AppError` carrying a code:

```cpp
enum class ErrorCode {
  ConfigInvalid,
  CameraOpenFailed,
  CameraFormatUnsupported,
  LowCaptureFps,
  FaceNotFound,
  QnnLibraryNotFound,
  QnnApiIncompatible,
  QnnGpuInitFailed,
  ModelManifestInvalid,
  ModelLoadFailed,
  InferenceFailed,
  OutputWriteFailed,
};

class AppError : public std::runtime_error {
 public:
  AppError(ErrorCode code, std::string message);
  ErrorCode code() const noexcept { return code_; }
 private:
  ErrorCode code_;
};

std::string_view to_string(ErrorCode code) noexcept;
```

Implement `to_string(ErrorCode)` as an exhaustive `switch` in `src/error.cpp`; an unknown value returns `UNKNOWN_ERROR`.

- [ ] **Step 3: Add `src/error.cpp` to `rppg_qnn_core` and run tests**

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add include/rppg_qnn/contracts.hpp include/rppg_qnn/error.hpp src/error.cpp CMakeLists.txt tests/test_contracts.cpp
git commit -m "feat: define runtime contracts and stable errors"
```

### Task 3: Implement strict CLI configuration

**Files:**
- Create: `include/rppg_qnn/config.hpp`
- Create: `src/config.cpp`
- Create: `tests/test_config.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write failing parser tests**

`tests/test_config.cpp` must cover these exact cases:

```cpp
auto camera = rppg_qnn::parse_config({"rppg_qnn_live", "--camera", "/dev/video2", "--fps", "30"});
EXPECT_EQ(camera.camera, "/dev/video2");
EXPECT_EQ(camera.fps, 30.0);

auto video = rppg_qnn::parse_config({"rppg_qnn_live", "--video", "sample.mp4", "--output", "outputs/a"});
EXPECT_EQ(video.video, "sample.mp4");

EXPECT_APP_ERROR(
  rppg_qnn::parse_config({"rppg_qnn_live", "--camera", "/dev/video0", "--video", "sample.mp4"}),
  rppg_qnn::ErrorCode::ConfigInvalid);
EXPECT_APP_ERROR(
  rppg_qnn::parse_config({"rppg_qnn_live", "--camera", "/dev/video0", "--backend", "cuda"}),
  rppg_qnn::ErrorCode::ConfigInvalid);
```

Add `EXPECT_APP_ERROR` to `test_support.hpp` so it catches `AppError` and checks its code.

- [ ] **Step 2: Implement the parser without a third-party CLI library**

Define:

```cpp
struct AppConfig {
  std::string camera;
  std::string video;
  int width{1280};
  int height{720};
  double fps{30.0};
  std::string traditional{"green"};
  std::string deep{"disabled"};
  std::string backend{"gpu"};
  std::string qnn_gpu_library{"libQnnGpu.so"};
  std::string opencl_library{"libOpenCL.so"};
  std::filesystem::path output{"outputs/session"};
  bool preflight_only{false};
};
```

`parse_config` accepts `--camera`, `--video`, `--width`, `--height`, `--fps`, `--traditional`, `--deep`, `--backend`, `--qnn-gpu-library`, `--opencl-library`, `--output`, and `--preflight-only`. `--deep` accepts only `disabled` or `fake` in this phase. Reject unknown flags, missing values, non-positive dimensions/FPS, unsupported algorithms/backends, and simultaneous camera/video. Default to `/dev/video0` only when neither source flag is provided and the command is not preflight-only.

- [ ] **Step 3: Register and run `test_config`**

Expected: all config cases pass.

- [ ] **Step 4: Commit**

```bash
git add include/rppg_qnn/config.hpp src/config.cpp tests/test_config.cpp tests/test_support.hpp CMakeLists.txt
git commit -m "feat: parse strict runtime configuration"
```

### Task 4: Add a thread-safe latest-only queue

**Files:**
- Create: `include/rppg_qnn/latest_queue.hpp`
- Create: `tests/test_latest_queue.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write overwrite, timeout, and close tests**

```cpp
rppg_qnn::LatestQueue<int> queue;
queue.push(1);
queue.push(2);
auto latest = queue.wait_pop(std::chrono::milliseconds(10));
EXPECT_TRUE(latest.has_value());
EXPECT_EQ(*latest, 2);
EXPECT_TRUE(!queue.wait_pop(std::chrono::milliseconds(1)).has_value());
queue.close();
EXPECT_TRUE(queue.closed());
EXPECT_TRUE(!queue.push(3));
```

- [ ] **Step 2: Implement `LatestQueue<T>`**

Use one `std::optional<T>`, mutex, condition variable and closed flag. `push(T)` replaces the pending value and never waits. `wait_pop(timeout)` moves out the pending value, resets it, and returns `nullopt` on timeout or closed-and-empty. `close()` is idempotent and wakes waiters.

- [ ] **Step 3: Add a 1000-iteration producer/consumer test**

Assert that the consumer receives monotonically increasing values and eventually receives the final value `999`; it is not required to receive all intermediate values.

- [ ] **Step 4: Build, run tests, and commit**

```bash
git add include/rppg_qnn/latest_queue.hpp tests/test_latest_queue.cpp CMakeLists.txt
git commit -m "feat: add non-blocking latest-only queue"
```

### Task 5: Implement video and V4L2 frame sources

**Files:**
- Create: `include/rppg_qnn/frame_source.hpp`
- Create: `src/frame_source.cpp`
- Create: `tests/test_frame_source.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a synthetic-video source test**

The test creates a temporary MJPG AVI with 12 frames at 24 FPS using `cv::VideoWriter`, opens it through `make_video_source`, and asserts:

```cpp
EXPECT_EQ(packet.frame_id, expected_id);
EXPECT_TRUE(packet.timestamp_sec > previous_timestamp);
EXPECT_EQ(packet.bgr.cols, 64);
EXPECT_EQ(packet.bgr.rows, 48);
```

After frame 12, `read()` returns `nullopt` and `eof()` is true.

- [ ] **Step 2: Define the interface**

```cpp
struct FramePacket {
  std::uint64_t frame_id{0};
  double timestamp_sec{0.0};
  cv::Mat bgr;
};

class FrameSource {
 public:
  virtual ~FrameSource() = default;
  virtual std::optional<FramePacket> read() = 0;
  virtual bool eof() const = 0;
  virtual double nominal_fps() const = 0;
};

std::unique_ptr<FrameSource> make_video_source(const std::filesystem::path& path);
std::unique_ptr<FrameSource> make_camera_source(const AppConfig& config);
```

- [ ] **Step 3: Implement sources**

Video timestamps are `frame_id / nominal_fps` and must remain strictly monotonic. Camera timestamps use `std::chrono::steady_clock` relative to the first successful frame. Open camera paths beginning with `/dev/video` through `cv::CAP_V4L2`; parse the numeric suffix and set width, height and FPS. Throw `CameraOpenFailed` for open/read failures before the first frame and `CameraFormatUnsupported` if negotiated dimensions are zero.

- [ ] **Step 4: Build, run tests, and commit**

```bash
git add include/rppg_qnn/frame_source.hpp src/frame_source.cpp tests/test_frame_source.cpp CMakeLists.txt
git commit -m "feat: read timestamped video and V4L2 frames"
```

### Task 6: Implement deterministic face and cheek ROI processing

**Files:**
- Create: `include/rppg_qnn/roi_processor.hpp`
- Create: `src/roi_processor.cpp`
- Create: `tests/test_roi_processor.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write geometry tests independent of the detector**

Given a 200×200 frame and face `{50, 40, 100, 120}`, assert `cheek_roi(face, frame_size)` returns an in-bounds rectangle covering the lower-middle face region and at least 5% of the face area. Assert negative/out-of-frame boxes are clipped and empty boxes return `nullopt`.

- [ ] **Step 2: Define ROI contracts**

```cpp
struct FaceBox { int x; int y; int width; int height; double confidence; };
struct RoiPacket {
  std::uint64_t frame_id{0};
  double timestamp_sec{0.0};
  cv::Mat roi_bgr;
  std::optional<FaceBox> face;
  bool used_fallback{false};
};

class RoiProcessor {
 public:
  explicit RoiProcessor(const std::filesystem::path& cascade_path);
  RoiPacket process(const FramePacket& frame);
};
```

- [ ] **Step 3: Implement the detector and fallback policy**

Use `cv::CascadeClassifier` for the first portable version. Detect every tenth frame; between detections reuse the last clipped face box. If no face exists, return an empty ROI and never substitute the entire frame. Convert the chosen cheek rectangle with `frame.bgr(rect).clone()` so packets are immutable after publication.

- [ ] **Step 4: Add an asset-path failure test**

Constructing with a missing cascade must throw `AppError(ErrorCode::ConfigInvalid, ...)` and include the missing path in the message.

- [ ] **Step 5: Build, run tests, and commit**

```bash
git add include/rppg_qnn/roi_processor.hpp src/roi_processor.cpp tests/test_roi_processor.cpp CMakeLists.txt
git commit -m "feat: extract stable cheek regions"
```

### Task 7: Implement traditional GREEN heart-rate estimation

**Files:**
- Create: `include/rppg_qnn/green_predictor.hpp`
- Create: `src/green_predictor.cpp`
- Create: `tests/test_green_predictor.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a 72 BPM synthetic-signal test**

Generate 20 seconds at jittered 27–31 FPS with green values `100 + 2*sin(2*pi*1.2*t)`, red/blue constants and fixed ROI quality. Feed timestamps and samples into `GreenPredictor`. Assert the final result is valid, method is `GREEN`, BPM is within 2 BPM of 72, source FPS is between 27 and 31, and waveform is non-empty.

- [ ] **Step 2: Write invalid-input tests**

Assert these exact invalid reasons:

- fewer than 8 seconds: `sampling`
- effective FPS below 15: `low_source_fps`
- maximum gap above 0.75 seconds: `capture_gap`
- flat green signal: `low_confidence`

- [ ] **Step 3: Implement the predictor**

`GreenPredictor::add_sample(timestamp, mean_bgr)` stores at most 30 seconds. Every second after an initial 10-second window it:

1. validates finite, strictly increasing timestamps;
2. computes effective FPS and maximum gap;
3. linearly resamples green to 30 Hz over the newest 10 seconds;
4. removes the mean and linear trend;
5. applies a Hann window;
6. computes a direct real DFT for 0.7–3.0 Hz bins;
7. selects the peak, converts to BPM, and computes confidence as peak power divided by total in-band power;
8. accepts only confidence ≥ 0.10 and BPM in 42–180.

The direct DFT is intentional for the first 300-sample window to avoid introducing another FFT dependency. Publish the normalized detrended waveform in `HeartRateResult`.

- [ ] **Step 4: Build, run tests, and commit**

```bash
git add include/rppg_qnn/green_predictor.hpp src/green_predictor.cpp tests/test_green_predictor.cpp CMakeLists.txt
git commit -m "feat: estimate traditional GREEN heart rate"
```

### Task 8: Add the deep-runtime boundary and asynchronous fake worker

**Files:**
- Create: `include/rppg_qnn/deep_runtime.hpp`
- Create: `include/rppg_qnn/deep_window_builder.hpp`
- Create: `include/rppg_qnn/deep_worker.hpp`
- Create: `src/deep_runtime.cpp`
- Create: `src/deep_window_builder.cpp`
- Create: `src/deep_worker.cpp`
- Create: `tests/test_deep_window_builder.cpp`
- Create: `tests/test_deep_worker.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the deep-window resampling test**

Feed 7 seconds of 72×72 RGB ROI frames with jittered 27–31 FPS timestamps to `DeepWindowBuilder(6.0, 180, {72, 72})`. Assert the first ready `DeepInput` has shape `{180, 72, 72, 3}`, strictly covers 6 seconds, contains finite RGB float values in row-major NHWC order, and reports the original source FPS and maximum gap. A source below 15 FPS must return status `low_source_fps`; a gap above 0.75 seconds must return `capture_gap`.

- [ ] **Step 2: Write a non-blocking scheduling test**

Use a fake runtime configured to sleep 100 ms per inference. Push 20 ready windows in under 20 ms. Assert `submit` never blocks longer than 5 ms, the worker produces at least one result, and the final processed window has the newest end timestamp.

- [ ] **Step 3: Define the runtime and window interfaces**

```cpp
struct DeepInput {
  double start_sec{0.0};
  double end_sec{0.0};
  double source_fps{0.0};
  double max_frame_gap_sec{0.0};
  std::vector<float> tensor;
  std::vector<std::int64_t> shape;
};

class IDeepRuntime {
 public:
  virtual ~IDeepRuntime() = default;
  virtual std::string backend_name() const = 0;
  virtual HeartRateResult infer(const DeepInput& input) = 0;
};

std::unique_ptr<IDeepRuntime> make_fake_deep_runtime(std::chrono::milliseconds latency);

class DeepWindowBuilder {
 public:
  DeepWindowBuilder(double window_sec, std::size_t sample_count, cv::Size input_size);
  std::optional<DeepInput> add_roi(const RoiPacket& packet);
  std::string status() const;
};
```

`DeepWorker` owns a `LatestQueue<DeepInput>`, one thread, the runtime and a mutex-protected latest result. `close()` closes the queue and joins once; its destructor calls `close()`.

- [ ] **Step 4: Implement the window builder**

Keep up to twice the requested window duration. Resize valid ROI frames to the configured size, convert BGR to RGB float32 without normalization, and preserve timestamps. Once the time span reaches six seconds, create 180 uniformly spaced target timestamps and choose the nearest source frame for each target. Reject non-monotonic timestamps, effective FPS below 15, gaps above 0.75 seconds and empty ROI frames. Store the tensor in NHWC order so the next plan can apply model-specific preprocessing without changing capture or scheduling.

- [ ] **Step 5: Implement deterministic fake inference**

The fake runtime requires shape `{N,H,W,3}`, computes the mean green value of every RGB frame, runs the same 0.7–3.0 Hz spectral estimator at 30 Hz, sets method `FAKE_DEEP`, backend `fake`, and records wall-clock inference time. Empty or malformed tensors return invalid reason `model_input_invalid`.

- [ ] **Step 6: Run the window and worker tests under sanitizers**

```bash
cmake -S . -B build-asan -DBUILD_TESTING=ON -DRPPG_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
```

Expected: no sanitizer reports.

- [ ] **Step 7: Commit**

```bash
git add include/rppg_qnn/deep_runtime.hpp include/rppg_qnn/deep_window_builder.hpp include/rppg_qnn/deep_worker.hpp src/deep_runtime.cpp src/deep_window_builder.cpp src/deep_worker.cpp tests/test_deep_window_builder.cpp tests/test_deep_worker.cpp CMakeLists.txt
git commit -m "feat: isolate asynchronous deep inference"
```

### Task 9: Implement QAIRT/QNN GPU runtime preflight

**Files:**
- Create: `include/rppg_qnn/qnn_preflight.hpp`
- Create: `src/qnn_preflight.cpp`
- Create: `tests/test_qnn_preflight.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write explicit-library probe tests**

Test a missing absolute path and assert `qnn_gpu_available == false` with `QNN_LIBRARY_NOT_FOUND` in the message. Probe the platform C library (resolved using `dladdr` in the test) as a stand-in and assert the handle opens. Probe a deliberately absent required symbol and assert `QNN_API_INCOMPATIBLE`.

- [ ] **Step 2: Implement safe probing**

```cpp
struct LibraryProbe {
  bool loaded{false};
  std::string resolved_path;
  std::string error;
};

LibraryProbe probe_library(const std::string& path, const std::vector<std::string>& required_symbols);
PreflightResult run_qnn_preflight(const AppConfig& config);
```

Use `dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL)`, clear/read `dlerror`, verify symbols with `dlsym`, and always `dlclose`. QNN GPU requires the SDK interface-provider symbol exposed by the installed headers; until the real SDK task pins that name, this phase only proves the library can load and reports symbol checking as `deferred_to_sdk_adapter`. OpenCL requires `clGetPlatformIDs`. `qnn_gpu_available` is true only when QNN GPU and OpenCL libraries both load.

- [ ] **Step 3: Test with environment-overridden paths**

Verify CLI values take precedence over `RPPG_QNN_GPU_LIBRARY` and `RPPG_OPENCL_LIBRARY`; defaults remain `libQnnGpu.so` and `libOpenCL.so`.

- [ ] **Step 4: Build, test, and commit**

```bash
git add include/rppg_qnn/qnn_preflight.hpp src/qnn_preflight.cpp tests/test_qnn_preflight.cpp CMakeLists.txt
git commit -m "feat: report QNN GPU runtime readiness"
```

### Task 10: Write terminal, JSONL and CSV result sinks

**Files:**
- Create: `include/rppg_qnn/result_sink.hpp`
- Create: `src/result_sink.cpp`
- Create: `tests/test_result_sink.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write exact schema tests**

Publish one preflight event and two heart-rate events to a temporary directory. Assert:

- `events.jsonl` has three parseable single-line objects;
- every line has `schema_version:1` and `event_type`;
- `heart_rate.csv` has the design-specified header and two data rows;
- commas, quotes and newlines in `invalid_reason` are escaped;
- `close()` writes `session_summary.json` atomically through a temporary file and rename.

The test may parse only the keys it needs using a small test helper; production serialization remains dependency-free.

- [ ] **Step 2: Define the sink interface**

```cpp
class ResultSink {
 public:
  explicit ResultSink(const std::filesystem::path& output_dir);
  void publish(const PreflightResult& result);
  void publish(const FrameHealth& health);
  void publish(const HeartRateResult& result);
  void close(int exit_code);
};
```

- [ ] **Step 3: Implement serialization**

Create the output directory before opening files. Implement `json_escape` for control characters, quotes and backslashes, and `csv_escape` using RFC 4180 quoting. Flush after every heart-rate result; batch frame-health events to at most one per second. Throw `OutputWriteFailed` if directory creation, open, write, flush or atomic rename fails.

- [ ] **Step 4: Build, test, and commit**

```bash
git add include/rppg_qnn/result_sink.hpp src/result_sink.cpp tests/test_result_sink.cpp CMakeLists.txt
git commit -m "feat: persist traceable runtime results"
```

### Task 11: Integrate the end-to-end pipeline and CLI

**Files:**
- Create: `include/rppg_qnn/pipeline.hpp`
- Create: `src/pipeline.cpp`
- Modify: `src/main.cpp`
- Create: `tests/test_pipeline.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write a full synthetic-video smoke test**

Generate a 16-second 30 FPS AVI whose cheek rectangle contains a 78 BPM green modulation. Inject a deterministic ROI processor in test mode so face detection does not depend on a cascade. Run `Pipeline` on the video and assert:

- exit code is zero;
- at least one valid GREEN result is within 3 BPM of 78;
- capture reaches the final frame;
- fake deep results exist without reducing processed frame count;
- JSONL and CSV files exist and are non-empty.

- [ ] **Step 2: Define injectable factories**

```cpp
struct PipelineDependencies {
  std::function<std::unique_ptr<FrameSource>()> make_source;
  std::function<std::unique_ptr<IRoiProcessor>()> make_roi;
  std::function<std::unique_ptr<IDeepRuntime>()> make_deep_runtime;
};

class Pipeline {
 public:
  Pipeline(AppConfig config, PipelineDependencies dependencies);
  int run();
};
```

Change `RoiProcessor` to implement `IRoiProcessor`. Production factories use camera/video, cascade ROI and fake deep runtime only when `--deep fake` is explicitly selected; the default is deep disabled until the QNN plan lands.

- [ ] **Step 3: Implement lifecycle order**

`run()` performs: parse/validate → output open → preflight → source open → ROI/deep-window/deep-worker creation → capture loop → worker close → sink close. Every valid ROI is sent to `DeepWindowBuilder`; each ready `DeepInput` is submitted to `DeepWorker` without waiting. Publish frame health once per second and terminal status once per second. On exceptions, publish a runtime error when the sink is available, close all workers, and map stable errors to nonzero exit codes.

- [ ] **Step 4: Replace `main.cpp`**

`main` converts `argv` to `std::vector<std::string>`, calls `parse_config`, builds production dependencies, runs the pipeline and prints one concise error line to `stderr` on `AppError` or `std::exception`.

- [ ] **Step 5: Build, run unit/integration tests, and commit**

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
git add include/rppg_qnn/pipeline.hpp src/pipeline.cpp src/main.cpp tests/test_pipeline.cpp CMakeLists.txt
git commit -m "feat: run the decoupled C++ rPPG pipeline"
```

### Task 12: Add reproducible build, smoke packaging and operator README

**Files:**
- Create: `scripts/build_linux.sh`
- Create: `scripts/run_smoke.sh`
- Create: `packaging/run_rppg_qnn.sh`
- Create: `README.md`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add install rules and verify staged install**

Add CMake install rules for `rppg_qnn_live`, default configs and README. Run:

```bash
cmake --install build --prefix stage/rppg-qnn
test -x stage/rppg-qnn/bin/rppg_qnn_live
```

Expected: executable is installed under `stage/rppg-qnn/bin` without models or Python files.

- [ ] **Step 2: Create native and cross-build scripts**

`scripts/build_linux.sh` must use `set -euo pipefail`, accept `native` or `aarch64`, reject unknown modes, configure Release plus tests, build, run host tests only for native mode, and stage install. AArch64 mode requires `AARCH64_TOOLCHAIN_PREFIX` and `AARCH64_SYSROOT` and passes the checked-in toolchain file.

- [ ] **Step 3: Create a controlled runtime launcher**

`packaging/run_rppg_qnn.sh` resolves its own directory, prepends only `${APP_ROOT}/lib` and optional `${QAIRT_TARGET_LIB_DIR}` to `LD_LIBRARY_PATH`, verifies `bin/rppg_qnn_live`, and executes it with unchanged arguments. It must not copy libraries into `/usr/lib` or use `sudo`.

- [ ] **Step 4: Write README operational instructions**

README must include:

- scope and non-medical disclaimer;
- proof that this repository is separate from `rPPG`;
- Ubuntu/OpenCV/CMake prerequisites;
- native and AArch64 build commands;
- QAIRT variables `QAIRT_SDK_ROOT` and `QAIRT_TARGET_LIB_DIR`;
- preflight command using explicit `libQnnGpu.so` and `libOpenCL.so` paths;
- V4L2 permission/device checks using `v4l2-ctl --list-devices`;
- video smoke command and live camera command;
- JSONL/CSV output descriptions;
- stable error-code troubleshooting table;
- statement that real deep output is disabled until the EfficientPhys QNN model plan is completed;
- deploy-by-version-directory and symlink rollback procedure.

- [ ] **Step 5: Run the complete verification matrix**

```bash
./scripts/build_linux.sh native
cmake -S . -B build-asan -DBUILD_TESTING=ON -DRPPG_ENABLE_SANITIZERS=ON
cmake --build build-asan --parallel
ctest --test-dir build-asan --output-on-failure
./build/rppg_qnn_live --preflight-only \
  --qnn-gpu-library /missing/libQnnGpu.so \
  --opencl-library /missing/libOpenCL.so \
  --output outputs/preflight-negative
```

Expected: native tests pass; sanitizer tests pass; negative preflight exits nonzero and records `QNN_LIBRARY_NOT_FOUND` rather than crashing.

- [ ] **Step 6: Verify repository isolation**

Run:

```bash
git -C /Users/wangjie/Documents/keti/rPPG status --short --branch
git status --short --branch
```

Expected: the original repository shows only changes that existed before this project; the new repository contains only planned C++ work.

- [ ] **Step 7: Commit**

```bash
git add CMakeLists.txt scripts packaging README.md
git commit -m "docs: package and operate the Linux runtime"
```

## Phase-1 completion gate

Before writing the EfficientPhys/QNN GPU implementation plan, verify all of the following:

- native Release and sanitizer tests pass;
- a generated 16-second video yields a bounded GREEN BPM result;
- fake deep latency cannot reduce capture frame count;
- missing QNN/OpenCL libraries produce stable preflight errors;
- an AArch64 package can be staged with no Python runtime files;
- the original `rPPG` repository remains untouched;
- the target-side QAIRT SDK headers, `libQnnGpu.so`, `libQnnSystem.so`, `libOpenCL.so`, exact library directories and QAIRT version have been collected for plan 2.
