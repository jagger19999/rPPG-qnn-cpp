# Android Camera2 Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real Android Camera2/`AImageReader` capture slice that enumerates cameras, requests permission, acquires timestamped `YUV_420_888` frames without Java frame copies, reports measured FPS, and stops deterministically without claiming rPPG or QNN results.

**Architecture:** Keep the Linux `FrameSource` and full OpenCV pipeline unchanged. Add a portable, host-tested YUV plane converter and an Android-only native camera session behind an opaque JNI handle. The Activity owns permission and lifecycle; native code owns Camera2, `AImageReader`, frame acquisition, timestamps, latest-frame dropping, and immutable status snapshots. OpenCV ROI/traditional processing remains the next plan.

**Tech Stack:** C++17, Android NDK Camera2, `AImageReader`, JNI, Java 17 Activity, Gradle 9.1.0, AGP 9.0.1, NDK 28.2.13676358, API 26+, `arm64-v8a`, host CTest.

---

## Scope and invariants

- The APK requests only `android.permission.CAMERA`.
- Java handles permission and lifecycle but never receives image buffers.
- Native capture requests `AIMAGE_FORMAT_YUV_420_888` and uses
  `AImageReader_acquireLatestImage`; stale frames are dropped by design.
- Plane row stride and pixel stride are validated before conversion.
- Sensor timestamps are converted from nanoseconds to seconds and must be
  strictly increasing for accepted frames.
- Stop and destroy are idempotent; no callback may touch a destroyed session.
- Status exposes camera ID, requested size/FPS, measured FPS, accepted/dropped
  frames, last timestamp, and stable error code/message.
- This slice does not import OpenCV Android, run ROI/POS/CHROM, write heart-rate
  output, import QNN, load a model, or expose fake inference.

## File map

### Portable files

- Create `include/rppg_qnn/yuv420.hpp`: non-owning YUV plane views, conversion
  result, and validation error type.
- Create `src/yuv420.cpp`: deterministic `YUV_420_888` to packed BGR conversion.
- Create `tests/test_yuv420.cpp`: contiguous, padded-row, interleaved-chroma,
  odd-dimension, and invalid-stride vectors.
- Modify `CMakeLists.txt`: compile the converter into the host core and register
  `test_yuv420`.
- Extend `include/rppg_qnn/error.hpp`, `src/error.cpp`, and
  `tests/test_contracts.cpp` with Android camera permission, unavailable ID,
  invalid image layout, and invalid native state identifiers.

### Android-only files

- Create `android/app/src/main/cpp/android_camera_session.hpp`: session config,
  immutable status, and RAII controller interface.
- Create `android/app/src/main/cpp/android_camera_session.cpp`: Camera2 device,
  capture request/session, `AImageReader`, latest-image callback, conversion,
  FPS accounting, and deterministic shutdown.
- Create `android/app/src/main/cpp/android_jni_handle.hpp`: synchronized opaque
  handle registry and state validation.
- Create `android/app/src/main/cpp/android_jni_handle.cpp`: create/start/stop/
  destroy/list/status operations without exceptions crossing JNI.
- Modify `android/app/src/main/cpp/native_bridge.cpp`: JNI string marshalling
  and calls into the handle layer.
- Modify `android/app/src/main/cpp/CMakeLists.txt`: compile portable conversion
  and Android camera sources; link `camera2ndk`, `mediandk`, `android`, and
  `log`.
- Modify `android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java`:
  native create/start/stop/destroy/list/status declarations.
- Modify `android/app/src/main/java/com/jagger/rppgbench/MainActivity.java`:
  permission request, camera list, start/stop controls, lifecycle stop, and
  low-rate status refresh.
- Modify `android/app/src/main/AndroidManifest.xml`: exact camera permission and
  non-required camera feature.
- Modify `android/app/src/main/res/values/strings.xml`: permission, controls,
  and explicit “camera smoke only” status text.
- Modify `tests/test_android_packaging.sh`: transition from “camera forbidden”
  to exact Camera2 source/permission/linkage requirements while keeping QNN,
  model, fake-deep, and non-`arm64-v8a` prohibitions.

## Task 1: Pin and validate the Gradle wrapper

**Files:**
- Modify: `tests/test_android_packaging.sh`
- Track: `android/gradlew`
- Track: `android/gradlew.bat`
- Track: `android/gradle/wrapper/gradle-wrapper.jar`
- Track: `android/gradle/wrapper/gradle-wrapper.properties`

- [ ] **Step 1: Extend the packaging contract and verify RED**

Require the four wrapper files in the tracked Android whitelist and assert:

```bash
grep -Fxq \
  'distributionUrl=https\://services.gradle.org/distributions/gradle-9.1.0-bin.zip' \
  "$root/android/gradle/wrapper/gradle-wrapper.properties"
grep -Fxq \
  'distributionSha256Sum=a17ddd85a26b6a7f5ddb71ff8b05fc5104c0202c6e64782429790c933686c806' \
  "$root/android/gradle/wrapper/gradle-wrapper.properties"
test "$(shasum -a 256 "$root/android/gradle/wrapper/gradle-wrapper.jar" |
  awk '{print $1}')" = \
  '76805e32c009c0cf0dd5d206bddc9fb22ea42e84db904b764f3047de095493f3'
```

Run:

```bash
ctest --test-dir build-android-full -R '^android_packaging$' --output-on-failure
```

Expected: fail until the tracked whitelist and wrapper checks agree.

- [ ] **Step 2: Make the wrapper contract GREEN**

Keep the official distribution URL and checksums above. `networkTimeout` may be
raised to `600000` for the current slow network, but the URL and checksums must
not be replaced by a mirror in tracked source.

- [ ] **Step 3: Verify**

```bash
ctest --test-dir build-android-full -R \
  '^(android_packaging|android_build_script)$' --output-on-failure
```

Expected: both tests pass.

## Task 2: Portable YUV_420_888 conversion contract

**Files:**
- Create: `include/rppg_qnn/yuv420.hpp`
- Create: `src/yuv420.cpp`
- Create: `tests/test_yuv420.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing converter tests**

Define the desired API in the test:

```cpp
struct YuvPlaneView {
  const std::uint8_t* data;
  std::size_t size;
  int row_stride;
  int pixel_stride;
};

struct Yuv420View {
  int width;
  int height;
  YuvPlaneView y;
  YuvPlaneView u;
  YuvPlaneView v;
};

std::vector<std::uint8_t> yuv420_to_bgr(const Yuv420View& image);
```

Use fixed BT.601 limited-range vectors and assert exact BGR bytes for:

1. planar 4x2 data with no padding;
2. padded Y and chroma rows;
3. pixel stride 2 interleaved chroma views;
4. odd 3x3 dimensions;
5. truncated planes, zero/negative stride, and pixel stride other than 1 or 2.

Run:

```bash
cmake -S . -B build-camera2-red -DCMAKE_BUILD_TYPE=Debug
cmake --build build-camera2-red --target test_yuv420 --parallel
```

Expected: compile failure because `rppg_qnn/yuv420.hpp` is absent.

- [ ] **Step 2: Implement minimal validated conversion**

For pixel `(x, y)`, sample Y at `y * y.row_stride + x * y.pixel_stride` and
sample U/V at `(y / 2) * row_stride + (x / 2) * pixel_stride`. Validate the
largest accessed offset against each plane size before conversion. Convert:

```cpp
const int c = static_cast<int>(Y) - 16;
const int d = static_cast<int>(U) - 128;
const int e = static_cast<int>(V) - 128;
const int r = clamp((298 * c + 409 * e + 128) >> 8);
const int g = clamp((298 * c - 100 * d - 208 * e + 128) >> 8);
const int b = clamp((298 * c + 516 * d + 128) >> 8);
```

Return packed BGR with exactly `width * height * 3` bytes. Throw
`std::invalid_argument` for invalid dimensions/layout and `std::overflow_error`
for size multiplication overflow.

- [ ] **Step 3: Verify RED to GREEN**

```bash
cmake --build build-camera2-red --target test_yuv420 --parallel
ctest --test-dir build-camera2-red -R '^yuv420$' --output-on-failure
```

Expected: pass with deterministic exact bytes.

## Task 3: Stable Android camera error identifiers

**Files:**
- Modify: `include/rppg_qnn/error.hpp`
- Modify: `src/error.cpp`
- Modify: `tests/test_contracts.cpp`

- [ ] **Step 1: Add failing contract assertions**

Add and assert stable text for:

```cpp
ErrorCode::CameraPermissionDenied   // CAMERA_PERMISSION_DENIED
ErrorCode::CameraIdUnavailable      // CAMERA_ID_UNAVAILABLE
ErrorCode::CameraImageInvalid       // CAMERA_IMAGE_INVALID
ErrorCode::NativeStateInvalid       // NATIVE_STATE_INVALID
```

Assign new exit codes after existing values; do not renumber existing errors.

- [ ] **Step 2: Run RED**

```bash
cmake --build build-camera2-red --target test_contracts --parallel
```

Expected: compile failure because the enum members do not exist.

- [ ] **Step 3: Implement mappings and run GREEN**

```bash
cmake --build build-camera2-red --target test_contracts --parallel
ctest --test-dir build-camera2-red -R '^contracts$' --output-on-failure
```

## Task 4: Android Camera2 session RAII

**Files:**
- Create: `android/app/src/main/cpp/android_camera_session.hpp`
- Create: `android/app/src/main/cpp/android_camera_session.cpp`
- Modify: `android/app/src/main/cpp/CMakeLists.txt`

- [ ] **Step 1: Add an Android source contract that fails**

Extend `tests/test_android_packaging.sh` to require:

```text
ACameraManager_getCameraIdList
AImageReader_new
AIMAGE_FORMAT_YUV_420_888
AImageReader_acquireLatestImage
AImage_getTimestamp
ACameraCaptureSession_stopRepeating
```

Require CMake linkage to `camera2ndk`, `mediandk`, `android`, and `log`.

Run the focused CTest and verify failure because the sources are absent.

- [ ] **Step 2: Implement enumeration and configuration**

`AndroidCameraSession::list_cameras()` creates an `ACameraManager`, retrieves
the ID list, copies every ID into owned strings, and always deletes the list and
manager. `start()` validates non-empty ID, positive even width/height, and
positive requested FPS before allocating native resources.

- [ ] **Step 3: Implement image acquisition**

Create an `AImageReader` with `AIMAGE_FORMAT_YUV_420_888` and a bounded
`maxImages` of 3. The callback calls `AImageReader_acquireLatestImage`, reads
timestamp and three planes/strides, converts with `yuv420_to_bgr`, updates only
owned status/counters under a mutex, and deletes the image on every path.
Conversion or camera failures set a stable error and stop accepting frames.

- [ ] **Step 4: Implement deterministic shutdown**

Shutdown order:

1. mark stopping under mutex;
2. remove the image listener;
3. stop repeating and abort captures;
4. close capture session;
5. free request, targets, output container, and output;
6. close camera device;
7. delete image reader;
8. delete camera manager;
9. publish stopped state.

Every pointer is nulled after release and repeated `stop()` is a no-op.

- [ ] **Step 5: Cross-build**

```bash
JAVA_HOME=/Users/wangjie/.local/jdks/zulu17.68.17-ca-jdk17.0.20-macosx_aarch64/Contents/Home \
ANDROID_SDK_ROOT=/opt/homebrew/share/android-commandlinetools \
ANDROID_HOME=/opt/homebrew/share/android-commandlinetools \
./scripts/build_android.sh
```

Expected: `assembleDebug` passes for `arm64-v8a`; no device success is claimed.

## Task 5: Opaque JNI lifecycle and immutable status

**Files:**
- Create: `android/app/src/main/cpp/android_jni_handle.hpp`
- Create: `android/app/src/main/cpp/android_jni_handle.cpp`
- Modify: `android/app/src/main/cpp/native_bridge.cpp`
- Modify: `android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java`

- [ ] **Step 1: Add failing packaging/API assertions**

Require these exact Java native methods:

```java
public static native String nativeListCameras();
public static native long nativeCreate(String cameraId, int width, int height,
                                       int fps);
public static native String nativeStart(long handle);
public static native String nativeStop(long handle);
public static native void nativeDestroy(long handle);
public static native String nativeGetStatus(long handle);
```

Run `android_packaging`; expected failure.

- [ ] **Step 2: Implement synchronized handle validation**

Use a registry mapping positive `jlong` IDs to `shared_ptr` session holders.
Never cast arbitrary Java integers to pointers. Lookup returns a strong
reference while the registry mutex is released. Destroy removes the registry
entry first, then stops the session. Invalid/zero handles return
`NATIVE_STATE_INVALID`; repeated destroy is harmless.

- [ ] **Step 3: Implement exception-safe JNI**

Every JNI entry is `noexcept`, validates null strings and conversion failures,
and maps exceptions to `CODE: message`. `nativeGetStatus` returns a compact
immutable JSON object containing only escaped owned strings and numeric fields.
No image byte array or direct image buffer is exposed to Java.

- [ ] **Step 4: Build and verify exported JNI symbols**

```bash
./scripts/build_android.sh
unzip -p android/app/build/outputs/apk/debug/app-debug.apk \
  lib/arm64-v8a/librppg_qnn_android.so > /tmp/librppg_qnn_android.so
"$ANDROID_SDK_ROOT/ndk/28.2.13676358/toolchains/llvm/prebuilt/darwin-x86_64/bin/llvm-nm" \
  -D /tmp/librppg_qnn_android.so
```

On Apple Silicon, use the actual NDK prebuilt directory returned by:

```bash
printf '%s\n' "$ANDROID_SDK_ROOT"/ndk/28.2.13676358/toolchains/llvm/prebuilt/*
```

Expected: all six JNI lifecycle symbols plus `nativeBuildIdentity` are present.

## Task 6: Permission, Activity lifecycle, and status UI

**Files:**
- Modify: `android/app/src/main/AndroidManifest.xml`
- Modify: `android/app/src/main/java/com/jagger/rppgbench/MainActivity.java`
- Modify: `android/app/src/main/res/values/strings.xml`
- Modify: `tests/test_android_packaging.sh`

- [ ] **Step 1: Write the failing manifest/UI contract**

Require exactly:

```xml
<uses-permission android:name="android.permission.CAMERA" />
<uses-feature android:name="android.hardware.camera.any"
              android:required="false" />
```

Require `requestPermissions`, `onRequestPermissionsResult`, `nativeListCameras`,
`nativeStart`, `nativeStop`, `nativeDestroy`, and `onStop`. Continue forbidding
storage/network/foreground-service permissions and all QNN/model/fake tokens.

- [ ] **Step 2: Implement permission-first UI**

On creation, show build identity and request camera permission only after the
user presses “List cameras” or “Start”. Parse no camera result as a visible
error. Use the first enumerated ID only as an explicit smoke default and display
it before start. Do not silently substitute another camera after a start error.

- [ ] **Step 3: Implement lifecycle**

Start creates one handle and starts it. Stop calls native stop but retains status
for display. `onStop()` stops capture. `onDestroy()` destroys the handle and
cancels status callbacks. Poll status at low rate on the UI thread only while
started; never block waiting for a frame.

- [ ] **Step 4: Build and inspect merged manifest**

```bash
./scripts/build_android.sh
"$ANDROID_SDK_ROOT/build-tools/36.0.0/aapt2" dump xmltree \
  android/app/build/outputs/apk/debug/app-debug.apk \
  AndroidManifest.xml
```

Expected: camera permission/optional feature only; launcher remains exported.

## Task 7: Host and APK acceptance for the Camera2 slice

**Files:**
- Modify: `README.md`
- Modify: `ANDROID_NEXT_STEPS.md`

- [ ] **Step 1: Run a fresh host suite**

```bash
cmake -S . -B build-camera2-release -DCMAKE_BUILD_TYPE=Release
cmake --build build-camera2-release --parallel
ctest --test-dir build-camera2-release --output-on-failure
```

Expected: all host and packaging tests pass.

- [ ] **Step 2: Build and audit APK**

```bash
./scripts/build_android.sh
unzip -l android/app/build/outputs/apk/debug/app-debug.apk
```

Expected: only `lib/arm64-v8a/librppg_qnn_android.so`; no model, QNN library,
checkpoint, dataset, or extra ABI.

- [ ] **Step 3: Record the device gate**

```bash
adb devices -l
adb install -r android/app/build/outputs/apk/debug/app-debug.apk
adb shell am start -n com.jagger.rppgbench/.MainActivity
adb shell dumpsys media.camera > media-camera.txt
```

If no authorized device is listed, record `DEVICE_NOT_CONNECTED` and stop the
device claim. Once connected, verify allow/deny permission paths, camera list,
strictly increasing timestamps, measured FPS, Activity stop/start, and 10
minutes of repeated start/stop without native crash or leaked camera ownership.

- [ ] **Step 4: Document exact boundary**

State that Camera2 capture is implemented and cross-built, while a real-device
claim requires retained ADB/logcat evidence. Explicitly state that ROI,
POS/CHROM, session files, QAIRT conversion, QNN, and Adreno remain subsequent
plans.

## Completion gate

This plan is complete only when:

1. all host tests pass after a real Android build directory exists;
2. the APK builds reproducibly for only `arm64-v8a`;
3. the manifest requests only camera permission;
4. YUV row/pixel stride conversion passes deterministic host vectors;
5. JNI list/create/start/stop/destroy/status is handle-safe and exception-safe;
6. Camera2 uses `AImageReader_acquireLatestImage` and deterministic shutdown;
7. no Java frame copy, OpenCV Android, QNN, model, or fake result is added; and
8. device behavior is either evidenced on an authorized device or explicitly
   reported as the remaining external gate.
