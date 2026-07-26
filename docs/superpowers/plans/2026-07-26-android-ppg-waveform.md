# Android PPG Waveform Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add independent traditional and TSCAN PPG waveform charts to the Android APK that reproduce the Python live waveform semantics without reducing camera or inference throughput.

**Architecture:** Publish immutable, revisioned waveform snapshots from the existing native result callback, expose them through a dedicated JNI float-array API, and render them with lightweight Android Canvas views. Existing status polling carries only revisions and progress; Java fetches waveform arrays only when a revision changes.

**Tech Stack:** C++17, Android NDK/JNI, Java 17, Android custom View/Canvas, JUnit 4, CTest.

---

## File map

- Modify `android/app/src/main/cpp/android_camera_session.hpp`: define public waveform snapshot metadata and retrieval API.
- Modify `android/app/src/main/cpp/android_camera_session.cpp`: normalize, publish, reset, and copy the two latest snapshots.
- Modify `android/app/src/main/cpp/android_jni_handle.hpp` and `.cpp`: expose snapshot metadata/values through handle-safe functions and status revisions.
- Modify `android/app/src/main/cpp/native_bridge.cpp`: convert native waveform values into Java `float[]` and metadata JSON.
- Modify `android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java`: declare waveform JNI calls.
- Create `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformSnapshot.java`: immutable Java waveform model and validation.
- Create `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformGeometry.java`: pure coordinate mapping for tests.
- Create `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformView.java`: allocation-conscious Canvas renderer.
- Create `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformCard.java`: chart title/state binding.
- Create `android/app/src/main/res/layout/view_ppg_waveform_card.xml`: reusable chart card.
- Modify `android/app/src/main/res/layout/activity_main.xml`: place both charts above camera preview.
- Modify `android/app/src/main/res/values/strings.xml`: chart titles and states.
- Modify `android/app/src/main/java/com/jagger/rppgbench/MainActivity.java`: revision polling, progress state, session clear/retain behavior.
- Add native and Java tests for normalization, publication, state mapping, geometry, and revision deduplication.

### Task 1: Native waveform snapshot contract

**Files:**
- Modify: `android/app/src/main/cpp/android_camera_session.hpp`
- Modify: `android/app/src/main/cpp/android_camera_session.cpp`
- Test: `android/app/src/main/cpp/android_camera_session_test.cpp`

- [ ] **Step 1: Write failing tests for normalization and revisioned publication**

Add tests proving that `[2, 3, 4]` becomes `[-1, 0, 1]`, non-finite and constant vectors are unavailable, traditional and deep revisions advance independently, and reset removes both snapshots.

- [ ] **Step 2: Run the native Android test target and verify RED**

Run:
```bash
cd android
./gradlew :app:externalNativeBuildDebug
```
Expected: compile failure because `WaveformSnapshot` and normalization helpers do not exist.

- [ ] **Step 3: Implement the native snapshot model**

Use this public shape:
```cpp
struct WaveformSnapshot {
  bool available{false};
  std::uint64_t revision{0};
  std::string method;
  double sample_rate_hz{0.0};
  bool is_valid{false};
  std::string invalid_reason;
  std::vector<float> values;
};
```
Publish a centered/max-absolute normalized copy from each `HeartRateResult` in the sink callback. Derive sample rate from `(source_frame_count - 1) / (window_end_sec - window_start_sec)` with `source_fps` fallback. Increment only the affected revision after a valid finite snapshot is ready.

- [ ] **Step 4: Verify native tests pass**

Run the new target and the existing CTest suite. Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: publish Android PPG waveform snapshots"
```

### Task 2: Dedicated JNI waveform transport

**Files:**
- Modify: `android/app/src/main/cpp/android_jni_handle.hpp`
- Modify: `android/app/src/main/cpp/android_jni_handle.cpp`
- Modify: `android/app/src/main/cpp/native_bridge.cpp`
- Modify: `android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java`
- Test: `android/app/src/test/java/com/jagger/rppgbench/WaveformNativeContractTest.java`

- [ ] **Step 1: Write failing Java contract tests**

Assert that status JSON carries `traditional_waveform_revision` and `deep_waveform_revision`, while waveform metadata and values are retrieved separately.

- [ ] **Step 2: Verify RED**

Run `./gradlew :app:testDebugUnitTest`. Expected: missing contract methods/fields.

- [ ] **Step 3: Implement JNI API**

Declare:
```java
public static native String nativeGetWaveformMetadata(long handle, boolean deep);
public static native float[] nativeGetWaveformValues(long handle, boolean deep);
```
Metadata contains availability, revision, method, sample rate, validity, invalid reason, and sample count. Return an empty array for unavailable data. Keep the existing status JSON free of waveform values.

- [ ] **Step 4: Verify Java tests and Android native build pass**

Run `./gradlew :app:testDebugUnitTest :app:externalNativeBuildDebug`. Expected: success.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: expose waveform snapshots through JNI"
```

### Task 3: Java waveform model and geometry

**Files:**
- Create: `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformSnapshot.java`
- Create: `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformGeometry.java`
- Test: `android/app/src/test/java/com/jagger/rppgbench/ui/PpgWaveformSnapshotTest.java`
- Test: `android/app/src/test/java/com/jagger/rppgbench/ui/PpgWaveformGeometryTest.java`

- [ ] **Step 1: Write failing model and geometry tests**

Cover defensive array copies, finite values, sample-rate validation, relative start label, and mapping first/last samples to left/right chart bounds with `-1/1` mapped to bottom/top.

- [ ] **Step 2: Verify RED**

Run the two named JUnit classes. Expected: missing classes.

- [ ] **Step 3: Implement minimal immutable model and pure mapper**

`PpgWaveformSnapshot` clones values on construction and exposure. `PpgWaveformGeometry.fillPoints(...)` writes interleaved x/y values into a caller-provided array and performs no allocation.

- [ ] **Step 4: Verify GREEN**

Run the named tests, then the full unit suite. Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/jagger/rppgbench/ui android/app/src/test/java/com/jagger/rppgbench/ui
git commit -m "feat: add Android PPG waveform model"
```

### Task 4: Native Canvas waveform cards

**Files:**
- Create: `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformView.java`
- Create: `android/app/src/main/java/com/jagger/rppgbench/ui/PpgWaveformCard.java`
- Create: `android/app/src/main/res/layout/view_ppg_waveform_card.xml`
- Modify: `android/app/src/main/res/layout/activity_main.xml`
- Modify: `android/app/src/main/res/values/strings.xml`
- Test: `android/app/src/test/java/com/jagger/rppgbench/ui/PpgWaveformCardStateTest.java`

- [ ] **Step 1: Write failing state formatter tests**

Verify exact disabled, sampling (`N / 180`), inference, invalid, and valid labels and green/orange color choices.

- [ ] **Step 2: Verify RED**

Run the named test. Expected: missing state formatter.

- [ ] **Step 3: Implement view and reusable card**

Preallocate paints and path, draw background/grid/zero line/polyline/endpoint labels, and invalidate only when `setSnapshot` receives a changed revision. Add two stacked includes between FPS and preview.

- [ ] **Step 4: Verify resources compile and tests pass**

Run `./gradlew :app:testDebugUnitTest :app:assembleDebug`. Expected: success.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main android/app/src/test
git commit -m "feat: render native PPG waveform cards"
```

### Task 5: MainActivity data binding and lifecycle

**Files:**
- Modify: `android/app/src/main/java/com/jagger/rppgbench/MainActivity.java`
- Test: `android/app/src/test/java/com/jagger/rppgbench/ui/PpgWaveformCoordinatorTest.java`

- [ ] **Step 1: Write failing coordinator tests**

Prove unchanged revisions do not fetch again, independent revisions update only their own card, new session clears both cards, stop retains both cards, and deep errors never clear traditional data.

- [ ] **Step 2: Verify RED**

Run the coordinator test. Expected: missing coordinator.

- [ ] **Step 3: Implement revision-based binding**

During the existing one-second status refresh, compare revisions, fetch metadata and values only for changes, validate sample counts, and bind each card. Clear revisions in `finishStartCamera` before native start; do not clear them in `stopCamera`.

- [ ] **Step 4: Verify full unit suite and APK build**

Run `./gradlew :app:testDebugUnitTest :app:assembleDebug`. Expected: success.

- [ ] **Step 5: Commit**

```bash
git commit -am "feat: bind live PPG waveforms in Android UI"
```

### Task 6: Regression and connected-device verification

**Files:**
- Modify only if a verification failure identifies a scoped defect.

- [ ] **Step 1: Run host and Android verification**

Run CTest plus `:app:testDebugUnitTest :app:assembleDebug`; require zero failures.

- [ ] **Step 2: Install the exact APK**

Run `adb install -r android/app/build/outputs/apk/debug/app-debug.apk`, record SHA-256 and package update time.

- [ ] **Step 3: Verify traditional-only mode**

Start portrait front-camera capture, wait for a complete traditional window, and capture a screenshot proving an upright preview and visible traditional waveform while deep says disabled.

- [ ] **Step 4: Verify TSCAN mode**

Enable a valid model, observe sampling/inference/complete states, and capture a screenshot with both independent waveforms. Record FPS before/after and reject a material sustained regression.

- [ ] **Step 5: Final repository hygiene**

Run `git diff --check`, confirm the worktree is clean, and report any external model prerequisite that prevents deep device verification.
