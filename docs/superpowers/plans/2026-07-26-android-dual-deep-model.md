# Android Selectable TSCAN and EfficientPhys Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a traditional-style deep-model selector for disabled, TSCAN, and EfficientPhys, with faithful per-model preprocessing, stage timings, model integrity gates, model-neutral waveforms, and transparent deep-result stabilization.

**Architecture:** Keep one asynchronous latest-only deep worker, but construct a model-specific runtime from an explicit enum. Shared `DeepInput` remains the uniform 180-frame RGB window; TSCAN and EfficientPhys transform it independently before separate ORT session adapters. Native result contracts carry raw/stabilized BPM and timing stages; Android chooses a model before capture and renders the selected model only.

**Tech Stack:** C++17, OpenCV, ONNX Runtime Android 1.27.0, Android NDK/JNI, Java 17, JUnit 4, CTest.

---

### Task 1: Deep model selection contract

**Files:**
- Modify: `android/app/src/main/cpp/android_camera_session.hpp`
- Modify: `android/app/src/main/cpp/android_camera_session.cpp`
- Modify: `android/app/src/main/cpp/android_jni_handle.hpp`
- Modify: `android/app/src/main/cpp/android_jni_handle.cpp`
- Modify: `android/app/src/main/cpp/native_bridge.cpp`
- Modify: `android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java`
- Test: `tests/test_config.cpp`

- [ ] Write failing tests for parsing `disabled`, `tscan`, and `efficientphys`, and rejecting every other value.
- [ ] Run the named CTest and confirm RED because no deep-model enum/parser exists.
- [ ] Add `DeepModel { Disabled, Tscan, EfficientPhys }`, canonical names, and replace the boolean configuration with model name plus path.
- [ ] Carry `deep_model` through JNI and status JSON while retaining derived `deep_enabled` for UI compatibility.
- [ ] Run focused tests and commit `feat: add explicit Android deep model selection`.

### Task 2: Restore the EfficientPhys runtime with exact preprocessing

**Files:**
- Create: `include/rppg_qnn/efficientphys_runtime.hpp`
- Create: `src/efficientphys_runtime.cpp`
- Modify: `CMakeLists.txt`
- Modify: `android/app/src/main/cpp/CMakeLists.txt`
- Test: `tests/test_efficientphys_runtime.cpp`

- [ ] Enable the existing EfficientPhys test target and verify it fails because the runtime is absent.
- [ ] Implement whole-window population mean/std normalization for `[180,72,72,3]`, TCHW conversion, duplicate final frame, and `[181,3,72,72]` output.
- [ ] Reuse the common waveform spectral postprocessor and return method `EFFICIENTPHYS` with all source-window metadata.
- [ ] Verify exact layout samples, invalid input/output behavior, finite waveform, and 72 BPM synthetic result.
- [ ] Run focused and full host tests; commit `feat: restore EfficientPhys runtime contract`.

### Task 3: Model-specific ONNX Runtime sessions and integrity gates

**Files:**
- Modify: `android/app/src/main/cpp/android_onnx_cpu_runtime.hpp`
- Modify: `android/app/src/main/cpp/android_onnx_cpu_runtime.cpp`
- Create: `android/app/src/main/java/com/jagger/rppgbench/ModelIntegrity.java`
- Test: `android/app/src/test/java/com/jagger/rppgbench/ModelIntegrityTest.java`

- [ ] Write failing Java tests for filenames and pinned SHA-256 values.
- [ ] Write compile-time/session tests for exact input and output names, types, and shapes for both models.
- [ ] Split the current TSCAN-only session into TSCAN `[180,6,72,72]` and EfficientPhys `[181,3,72,72]` adapters; construct the matching shared runtime.
- [ ] Validate filename and SHA-256 in Java before native configuration; native ORT shape inspection remains the second gate.
- [ ] Run Android tests/build and commit `feat: run selectable deep ONNX models`.

### Task 4: Stage-level timing contract

**Files:**
- Modify: `include/rppg_qnn/contracts.hpp`
- Modify: `include/rppg_qnn/deep_runtime.hpp`
- Modify: `src/deep_window_builder.cpp`
- Modify: `src/tscan_runtime.cpp`
- Modify: `src/efficientphys_runtime.cpp`
- Modify: `android/app/src/main/cpp/android_camera_session.hpp`
- Modify: `android/app/src/main/cpp/android_camera_session.cpp`
- Modify: `android/app/src/main/cpp/android_jni_handle.cpp`
- Modify: `src/result_sink.cpp`
- Test: `tests/test_tscan_runtime.cpp`
- Test: `tests/test_efficientphys_runtime.cpp`
- Test: `tests/test_result_sink.cpp`

- [ ] Write failing tests requiring finite non-negative materialization, preprocessing, runtime, postprocessing, and total timing fields.
- [ ] Add timing fields without changing the existing `inference_ms` meaning.
- [ ] Measure ORT `Session::Run` inside the session adapter; measure other stages at their actual boundaries.
- [ ] Export every field in status JSON, result JSONL, and CSV.
- [ ] Run tests and commit `feat: report deep inference stage timings`.

### Task 5: Transparent deep BPM stabilization

**Files:**
- Create: `include/rppg_qnn/deep_stabilizer.hpp`
- Create: `src/deep_stabilizer.cpp`
- Modify: `include/rppg_qnn/contracts.hpp`
- Modify: `src/tscan_runtime.cpp`
- Modify: `src/efficientphys_runtime.cpp`
- Test: `tests/test_deep_stabilizer.cpp`

- [ ] Write failing tests for low-confidence rejection, supported harmonic selection, unsupported jump rejection, three-value median, and reset.
- [ ] Implement a stateful stabilizer using only the selected deep model's waveform/results; never consume traditional or watch BPM.
- [ ] Preserve `raw_bpm`; publish `display_bpm`, correction reason, and stability validity separately.
- [ ] Reset stabilizer when a runtime/session/model changes.
- [ ] Run tests and commit `feat: stabilize deep BPM transparently`.

### Task 6: Android selector and model-neutral cards

**Files:**
- Modify: `android/app/src/main/res/layout/activity_main.xml`
- Modify: `android/app/src/main/res/values/strings.xml`
- Modify: `android/app/src/main/java/com/jagger/rppgbench/MainActivity.java`
- Modify: `android/app/src/main/java/com/jagger/rppgbench/ui/HrStatusFormatter.java`
- Modify: `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformState.java`
- Test: `android/app/src/test/java/com/jagger/rppgbench/DeepModelSelectionTest.java`
- Test: `android/app/src/test/java/com/jagger/rppgbench/ui/HrStatusFormatterTest.java`
- Test: `android/app/src/test/java/com/jagger/rppgbench/ui/PpgWaveformStateTest.java`

- [ ] Write failing tests mapping the three spinner values to model name/path and model-specific labels.
- [ ] Replace the checkbox with a spinner, default to `关闭`, and disable both traditional/deep selectors while capture is active.
- [ ] Show selected model, stabilized BPM, raw BPM, confidence, total/stage timings, and model-neutral waveform state.
- [ ] Clear deep history on a new session but retain the last result after stop.
- [ ] Run unit tests/build and commit `feat: select deep model in Android UI`.

### Task 7: Allocation and thread benchmark preparation

**Files:**
- Modify: `src/tscan_preprocessor.cpp`
- Modify: `src/efficientphys_runtime.cpp`
- Modify: `android/app/src/main/cpp/android_onnx_cpu_runtime.cpp`
- Test: `tests/test_tscan_preprocessor.cpp`
- Test: `tests/test_efficientphys_runtime.cpp`

- [ ] Add tests proving repeated calls preserve exact outputs.
- [ ] Reuse worker-local scratch buffers where this removes repeated large allocations without changing tensor values.
- [ ] Make ORT intra-op thread count an internal validated constructor option; retain inter-op count 1.
- [ ] Build three instrumented APKs for 2/4/6 threads and record sustained model runtime, camera FPS, dropped frames, and device temperature for each model.
- [ ] Select the fastest non-regressing default and commit `perf: reduce deep runtime overhead`.

### Task 8: Full regression and connected-device validation

**Files:**
- Modify: `tests/test_android_packaging.sh` only for intentional source additions.

- [ ] Run the full host CTest suite and Android unit/build tasks with zero failures.
- [ ] Verify both app-private model SHA-256 values before starting the app.
- [ ] Install the exact APK and test `关闭`, `TSCAN`, and `EfficientPhys` sessions independently.
- [ ] Capture screenshots proving upright portrait preview, correct model card/waveform labels, and 180-point waveforms.
- [ ] Record stage timings, sustained FPS, dropped frames, memory, raw/stabilized BPM, confidence, and invalid reason.
- [ ] Run a paired watch session for each model and export per-model MAE/invalid-rate evidence; if the watch is unavailable, report this gate as pending rather than claiming accuracy.
- [ ] Run `git diff --check`, verify the branch is clean, and report the installed APK SHA-256 and package update time.
