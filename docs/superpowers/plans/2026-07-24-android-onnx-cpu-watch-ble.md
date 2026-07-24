# Android ORT CPU + Watch BLE Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the phone-demo APK so GREEN/POS/CHROM and EfficientPhys (ONNX Runtime CPU) run from Camera2, and HUAWEI GT 5 Pro heart-rate broadcast is received over BLE as an experimental reference aligned to rPPG windows.

**Architecture:** Keep camera/rPPG/deep in the existing native pipeline. Import the already-validated external ONNX into app-private storage. Implement watch parse/store/align/worker as pure Java (+ Android BLE adapter), decoupled from camera stop. Merge watch metrics into the UI status poll; write watch/alignment CSVs into the session directory.

**Tech Stack:** Android API 26+, Camera2 NDK, OpenCV 4.13.0 Android, ONNX Runtime Android 1.27.0 CPU, Java BLE GATT (`0x180D`/`0x2A37`), CMake/NDK 28.2, Gradle AGP 9.0.1, host CTest, Android JVM unit tests (JUnit 4).

**Spec:** `docs/superpowers/specs/2026-07-24-android-onnx-cpu-watch-ble-design.md`

**Worktree:** `/Users/wangjie/.config/superpowers/worktrees/rPPG-qnn-cpp/android-ndk-runtime`

**Pinned artifacts (do not commit into Git):**
- ONNX: `/Users/wangjie/.config/superpowers/worktrees/rPPG-qnn-cpp/efficientphys-qnn-export/artifacts/model_export/efficientphys_pure/efficientphys_pure.onnx`  
  SHA-256 `c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d`
- ORT root env: `RPPG_ONNXRUNTIME_ANDROID` → verified 1.27.0 extract  
  AAR SHA-256 `077dec5e2d821234c7dc0aba584bec8f999854b546c754cab93a90741c56fbeb`
- OpenCV env: `RPPG_OPENCV_ANDROID_SDK`

---

## File map

| Path | Responsibility |
|------|----------------|
| `android/app/src/main/java/com/jagger/rppgbench/watch/HeartRateParser.java` | Pure 0x2A37 parser |
| `android/app/src/main/java/com/jagger/rppgbench/watch/WatchContracts.java` | Enums + immutable samples/snapshots/alignment |
| `android/app/src/main/java/com/jagger/rppgbench/watch/WatchSampleStore.java` | Thread-safe 180s history + status |
| `android/app/src/main/java/com/jagger/rppgbench/watch/WatchAligner.java` | Window alignment + summary |
| `android/app/src/main/java/com/jagger/rppgbench/watch/WatchBleBackend.java` | Interface over Android BLE |
| `android/app/src/main/java/com/jagger/rppgbench/watch/AndroidBleBackend.java` | Scanner + GATT notify |
| `android/app/src/main/java/com/jagger/rppgbench/watch/WatchBleWorker.java` | Scan/connect/reconnect state machine |
| `android/app/src/main/java/com/jagger/rppgbench/watch/WatchCsvExport.java` | Session CSV writers |
| `android/app/src/test/java/com/jagger/rppgbench/watch/*Test.java` | JVM unit tests |
| `android/app/src/main/java/.../MainActivity.java` | Watch UI + permissions + status merge |
| `android/app/src/main/AndroidManifest.xml` | BLE permissions/features |
| `android/app/build.gradle` | `testOptions` + JUnit dependency |
| `scripts/import_efficientphys_model.sh` | Host-side hash check + adb import helper |
| `scripts/build_android.sh` | Keep ORT/OpenCV gates |
| `ANDROID_NEXT_STEPS.md` / `README.md` | Phone commands + remaining gates |
| Existing native ORT files | Already present uncommitted; stabilize & verify |

Python behavioral reference (read-only):  
`/Users/wangjie/Documents/keti/rPPG/src/rppg_cabin/watch/{parser,store,alignment,worker,contracts}.py`

---

### Task 1: Stabilize existing ORT CPU native path (host + APK)

**Files:**
- Existing uncommitted: `android/app/src/main/cpp/android_onnx_cpu_runtime.*`, session/JNI/CMake/Gradle changes
- Modify if needed: `tests/test_android_packaging.sh`, `scripts/build_android.sh`
- Verify: host CTest + `./scripts/build_android.sh`

- [ ] **Step 1: Inventory uncommitted ORT/camera work**

```bash
cd /Users/wangjie/.config/superpowers/worktrees/rPPG-qnn-cpp/android-ndk-runtime
git status --short
test -f android/app/src/main/cpp/android_onnx_cpu_runtime.cpp
test -f /Users/wangjie/.local/android-deps/onnxruntime-android-1.27.0/headers/onnxruntime_cxx_api.h
```

Expected: ORT sources exist; ORT SDK installed locally.

- [ ] **Step 2: Run host tests**

```bash
cmake --build build-camera2-release --parallel
ctest --test-dir build-camera2-release --output-on-failure
```

Expected: all tests PASS. If `android_packaging` fails, fix gates to allow ORT CPU (still forbid QNN/fake and forbid `.onnx` under `android/`).

- [ ] **Step 3: Cross-build APK**

```bash
export JAVA_HOME="/Users/wangjie/.local/jdks/zulu17.68.17-ca-jdk17.0.20-macosx_aarch64/Contents/Home"
export ANDROID_SDK_ROOT="/opt/homebrew/share/android-commandlinetools"
export ANDROID_HOME="$ANDROID_SDK_ROOT"
export RPPG_OPENCV_ANDROID_SDK="/Users/wangjie/.local/android-deps/OpenCV-android-sdk"
export RPPG_ONNXRUNTIME_ANDROID="/Users/wangjie/.local/android-deps/onnxruntime-android-1.27.0"
./scripts/build_android.sh
unzip -l android/app/build/outputs/apk/debug/app-debug.apk | rg 'libonnxruntime|librppg|haarcascade|\.onnx'
```

Expected: APK path printed; contains `libonnxruntime.so` + `librppg_qnn_android.so` + Haar asset; **no** `.onnx`.

- [ ] **Step 4: Commit stabilized ORT/camera native slice**

```bash
git add android/app/src/main/cpp/android_onnx_cpu_runtime.cpp \
  android/app/src/main/cpp/android_onnx_cpu_runtime.hpp \
  android/app/src/main/cpp/CMakeLists.txt \
  android/app/build.gradle \
  android/app/src/main/cpp/android_camera_session.cpp \
  android/app/src/main/cpp/android_camera_session.hpp \
  android/app/src/main/cpp/android_jni_handle.cpp \
  android/app/src/main/cpp/android_jni_handle.hpp \
  android/app/src/main/cpp/android_qnn_preflight_stub.cpp \
  android/app/src/main/cpp/native_bridge.cpp \
  android/app/src/main/java/com/jagger/rppgbench/MainActivity.java \
  android/app/src/main/java/com/jagger/rppgbench/NativeBridge.java \
  android/app/src/main/AndroidManifest.xml \
  android/app/src/main/res/values/strings.xml \
  android/app/src/main/assets/haarcascade_frontalface_default.xml \
  android/gradlew android/gradlew.bat android/gradle/wrapper \
  scripts/build_android.sh tests/test_android_*.sh \
  src/pipeline.cpp src/deep_worker.cpp src/build_identity.cpp \
  src/qnn_preflight.cpp src/error.cpp include/rppg_qnn/error.hpp \
  include/rppg_qnn/yuv420.hpp src/yuv420.cpp tests/test_yuv420.cpp \
  tests/test_contracts.cpp tests/test_deep_worker.cpp \
  CMakeLists.txt README.md ANDROID_NEXT_STEPS.md
# Only add files that exist and belong to this slice; do not add .onnx/.pth
git commit -m "$(cat <<'EOF'
feat: ship Android ORT CPU EfficientPhys beside traditional rPPG

Cross-build Camera2 + OpenCV GREEN/POS/CHROM with an explicit
ONNX Runtime CPU backend and external model import boundary.
EOF
)"
```

---

### Task 2: Model import helper + frozen-vector host check

**Files:**
- Create: `scripts/import_efficientphys_model.sh`
- Create or finish: `tests/test_efficientphys_runtime.cpp` (host-side contract using existing npy if available via env)
- Modify: `ANDROID_NEXT_STEPS.md` model import section (if missing exact commands)

- [ ] **Step 1: Write import script**

Create `scripts/import_efficientphys_model.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
die() { printf 'import_efficientphys_model.sh: %s\n' "$*" >&2; exit 2; }
MODEL=${1:-}
EXPECTED=c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d
PACKAGE=com.jagger.rppgbench
[[ -n "$MODEL" && -f "$MODEL" ]] || die "usage: $0 /absolute/path/to/efficientphys_pure.onnx"
actual=$(shasum -a 256 "$MODEL" | awk '{print $1}')
[[ "$actual" == "$EXPECTED" ]] || die "checksum mismatch: $actual"
adb get-state >/dev/null || die "no adb device"
adb push "$MODEL" /data/local/tmp/efficientphys_pure.onnx
adb shell run-as "$PACKAGE" mkdir -p files/models
adb shell run-as "$PACKAGE" cp /data/local/tmp/efficientphys_pure.onnx files/models/efficientphys_pure.onnx
adb shell run-as "$PACKAGE" sha256sum files/models/efficientphys_pure.onnx
adb shell rm /data/local/tmp/efficientphys_pure.onnx
printf 'imported into %s files/models/efficientphys_pure.onnx\n' "$PACKAGE"
```

```bash
chmod +x scripts/import_efficientphys_model.sh
```

- [ ] **Step 2: Verify local ONNX hash without adb**

```bash
MODEL="/Users/wangjie/.config/superpowers/worktrees/rPPG-qnn-cpp/efficientphys-qnn-export/artifacts/model_export/efficientphys_pure/efficientphys_pure.onnx"
test "$(shasum -a 256 "$MODEL" | awk '{print $1}')" = c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d
```

Expected: exit 0.

- [ ] **Step 3: Add packaging assertion that import script exists and pins the hash**

In `tests/test_android_packaging.sh`, after other checks:

```bash
import_script="$root/scripts/import_efficientphys_model.sh"
if [[ ! -x "$import_script" ]] ||
   ! grep -Fq 'c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d' "$import_script"; then
  echo "android packaging check: EfficientPhys import helper/hash missing" >&2
  exit 1
fi
```

- [ ] **Step 4: Run packaging test and commit**

```bash
ctest --test-dir build-camera2-release -R 'android_packaging' --output-on-failure
git add scripts/import_efficientphys_model.sh tests/test_android_packaging.sh ANDROID_NEXT_STEPS.md
git commit -m "chore: add EfficientPhys ONNX adb import helper"
```

---

### Task 3: Enable Android JVM unit tests

**Files:**
- Modify: `android/app/build.gradle`
- Create: `android/app/src/test/java/com/jagger/rppgbench/watch/HeartRateParserTest.java` (first failing test in Task 4)

- [ ] **Step 1: Add JUnit dependency**

In `android/app/build.gradle`, append:

```gradle
dependencies {
    testImplementation 'junit:junit:4.13.2'
}
```

Ensure `android { ... }` already has Java 17. No AndroidX migration required.

- [ ] **Step 2: Confirm unit test task exists**

```bash
export JAVA_HOME="/Users/wangjie/.local/jdks/zulu17.68.17-ca-jdk17.0.20-macosx_aarch64/Contents/Home"
export ANDROID_SDK_ROOT="/opt/homebrew/share/android-commandlinetools"
android/gradlew --no-daemon --project-dir android :app:tasks --all | rg 'testDebugUnitTest|test'
```

Expected: `testDebugUnitTest` listed.

- [ ] **Step 3: Commit gradle test wiring**

```bash
git add android/app/build.gradle
git commit -m "build: enable Android JVM unit tests for watch module"
```

---

### Task 4: Heart rate measurement parser (TDD)

**Files:**
- Create: `android/app/src/main/java/com/jagger/rppgbench/watch/HeartRateParser.java`
- Create: `android/app/src/test/java/com/jagger/rppgbench/watch/HeartRateParserTest.java`

- [ ] **Step 1: Write failing tests**

```java
package com.jagger.rppgbench.watch;

import org.junit.Test;
import static org.junit.Assert.*;

public final class HeartRateParserTest {
    @Test public void parse8BitBpm() {
        HeartRateParser.Parsed parsed = HeartRateParser.parse(new byte[] {0x00, 72});
        assertEquals(72, parsed.bpm);
        assertEquals(0, parsed.rrIntervalsSec.length);
    }

    @Test public void parse16BitBpmAndRr() {
        HeartRateParser.Parsed parsed =
                HeartRateParser.parse(new byte[] {0x11, 44, 1, 0, 4, 32, 4});
        assertEquals(300, parsed.bpm);
        assertEquals(2, parsed.rrIntervalsSec.length);
        assertEquals(1.0, parsed.rrIntervalsSec[0], 1e-9);
        assertEquals(1.03125, parsed.rrIntervalsSec[1], 1e-9);
    }

    @Test public void parseEnergyExpendedBeforeRr() {
        HeartRateParser.Parsed parsed =
                HeartRateParser.parse(new byte[] {0x18, 72, 10, 0, 0, 4});
        assertEquals(72, parsed.bpm);
        assertEquals(1.0, parsed.rrIntervalsSec[0], 1e-9);
    }

    @Test(expected = IllegalArgumentException.class)
    public void rejectTruncated() {
        HeartRateParser.parse(new byte[] {0x01, 0x48});
    }
}
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest --tests com.jagger.rppgbench.watch.HeartRateParserTest
```

Expected: FAIL (class missing).

- [ ] **Step 3: Implement parser**

Port logic from `/Users/wangjie/Documents/keti/rPPG/src/rppg_cabin/watch/parser.py`:

```java
package com.jagger.rppgbench.watch;

public final class HeartRateParser {
    public static final class Parsed {
        public final int bpm;
        public final double[] rrIntervalsSec;
        public Parsed(int bpm, double[] rrIntervalsSec) {
            this.bpm = bpm;
            this.rrIntervalsSec = rrIntervalsSec;
        }
    }

    private HeartRateParser() {}

    public static Parsed parse(byte[] data) {
        if (data == null || data.length < 2) {
            throw new IllegalArgumentException("heart rate packet is truncated");
        }
        int flags = data[0] & 0xFF;
        int cursor = 1;
        int bpm;
        if ((flags & 0x01) != 0) {
            if (data.length < 3) {
                throw new IllegalArgumentException("16-bit heart rate packet is truncated");
            }
            bpm = (data[cursor] & 0xFF) | ((data[cursor + 1] & 0xFF) << 8);
            cursor += 2;
        } else {
            bpm = data[cursor] & 0xFF;
            cursor += 1;
        }
        if ((flags & 0x08) != 0) {
            cursor += 2;
            if (cursor > data.length) {
                throw new IllegalArgumentException("energy expended field is truncated");
            }
        }
        double[] rr = new double[0];
        if ((flags & 0x10) != 0) {
            if (((data.length - cursor) & 1) != 0) {
                throw new IllegalArgumentException("RR interval field is truncated");
            }
            int count = (data.length - cursor) / 2;
            rr = new double[count];
            for (int i = 0; i < count; i++) {
                int raw = (data[cursor] & 0xFF) | ((data[cursor + 1] & 0xFF) << 8);
                cursor += 2;
                rr[i] = raw / 1024.0;
            }
        }
        return new Parsed(bpm, rr);
    }
}
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest --tests com.jagger.rppgbench.watch.HeartRateParserTest
```

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/jagger/rppgbench/watch/HeartRateParser.java \
  android/app/src/test/java/com/jagger/rppgbench/watch/HeartRateParserTest.java
git commit -m "feat: parse BLE Heart Rate Measurement packets"
```

---

### Task 5: Watch contracts + sample store (TDD)

**Files:**
- Create: `android/app/src/main/java/com/jagger/rppgbench/watch/WatchContracts.java`
- Create: `android/app/src/main/java/com/jagger/rppgbench/watch/WatchSampleStore.java`
- Create: `android/app/src/test/java/com/jagger/rppgbench/watch/WatchSampleStoreTest.java`

- [ ] **Step 1: Write failing store tests**

Cover at least:
- Accept measurement → latest BPM available, status `STREAMING`
- History bounded to 180 seconds (drop older samples when accepting newer ones)
- `isStale(now)` true when latest sample older than 2.0s → snapshot status becomes `STALE` on `snapshot(now)`
- BPM outside `[1,300]` rejected / not stored
- Malformed counter increments without wiping history

Reference behavior: `/Users/wangjie/Documents/keti/rPPG/tests/test_watch_store.py`

- [ ] **Step 2: Run — FAIL**

```bash
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest --tests com.jagger.rppgbench.watch.WatchSampleStoreTest
```

- [ ] **Step 3: Implement contracts + store**

`WatchContracts.java` must define:
- `WatchConnectionStatus`: DISCONNECTED, SCANNING, CONNECTING, STREAMING, STALE, RECONNECTING, ERROR
- `WatchAlignmentStatus`: PENDING, ALIGNED, PARTIAL_COVERAGE, WATCH_STALE, RPPG_INVALID, DISCONNECTED
- Immutable `WatchDevice`, `WatchHeartRateSample`, `WatchHeartRateSnapshot`, `WatchAlignmentResult`

`WatchSampleStore`:
- `synchronized` mutation
- `acceptMeasurement(monotonicSec, bpm, rr, deviceId, deviceName)`
- `setStatus(...)`, `setDevices(...)`, `setError(...)`, `incrementMalformed()`, `setReconnectAttempts(...)`
- `snapshot(nowMonotonicSec)` copies samples, applies STALE if streaming and gap > 2s
- Retain samples with `received_monotonic_sec >= now - 180` (use latest sample time or explicit now)

- [ ] **Step 4: Run — PASS, then commit**

```bash
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest --tests 'com.jagger.rppgbench.watch.*'
git add android/app/src/main/java/com/jagger/rppgbench/watch/WatchContracts.java \
  android/app/src/main/java/com/jagger/rppgbench/watch/WatchSampleStore.java \
  android/app/src/test/java/com/jagger/rppgbench/watch/WatchSampleStoreTest.java
git commit -m "feat: add watch sample store and contracts"
```

---

### Task 6: Window aligner (TDD)

**Files:**
- Create: `android/app/src/main/java/com/jagger/rppgbench/watch/WatchAligner.java`
- Create: `android/app/src/test/java/com/jagger/rppgbench/watch/WatchAlignerTest.java`

- [ ] **Step 1: Write failing alignment tests**

Port cases from `/Users/wangjie/Documents/keti/rPPG/tests/test_watch_alignment.py`:
- DISCONNECTED / STALE / invalid rPPG → matching status, no formal error
- <3 samples or coverage <0.7 or max gap >2 → `PARTIAL_COVERAGE`
- Valid window → median BPM, signed/absolute error, `ALIGNED`

Helper input type (local to aligner API):

```java
public static final class RppgWindow {
    public final double startSec;
    public final double endSec;
    public final Double bpm; // null if unavailable
    public final boolean valid;
    public RppgWindow(double startSec, double endSec, Double bpm, boolean valid) {
        this.startSec = startSec;
        this.endSec = endSec;
        this.bpm = bpm;
        this.valid = valid;
    }
}
```

- [ ] **Step 2: Implement `WatchAligner.align(RppgWindow, WatchHeartRateSnapshot, double sessionStartMonotonicSec)`**

Median: sort BPM ints and take middle (average of two middles if even — match Python `numpy.median`).

- [ ] **Step 3: Also implement `summarize(List<WatchAlignmentResult>)` returning MAE/RMSE/bias/valid ratio for ALIGNED only.**

- [ ] **Step 4: Run tests PASS and commit**

```bash
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest --tests com.jagger.rppgbench.watch.WatchAlignerTest
git add android/app/src/main/java/com/jagger/rppgbench/watch/WatchAligner.java \
  android/app/src/test/java/com/jagger/rppgbench/watch/WatchAlignerTest.java
git commit -m "feat: align watch reference BPM to rPPG windows"
```

---

### Task 7: BLE backend interface + worker state machine (TDD with fake backend)

**Files:**
- Create: `WatchBleBackend.java`, `WatchBleWorker.java`
- Create: `WatchBleWorkerTest.java` with `FakeBleBackend`
- Create later: `AndroidBleBackend.java` (Task 8)

- [ ] **Step 1: Define backend interface**

```java
package com.jagger.rppgbench.watch;

import java.util.List;
import java.util.function.Consumer;

public interface WatchBleBackend {
    interface DeviceHandle {
        String id();
        String name();
    }

    List<DeviceHandle> scan(double timeoutSec) throws Exception;

    void connect(
            DeviceHandle handle,
            Consumer<byte[]> onNotification,
            Runnable onDisconnect) throws Exception;

    void disconnect() throws Exception;
}
```

- [ ] **Step 2: Write worker tests with FakeBleBackend**

Required behaviors (mirror Python worker):
- `startScan(15)` → status SCANNING then devices populated / back to DISCONNECTED
- `connect(deviceId)` → CONNECTING → STREAMING on notify
- Notify payload parsed into store with monotonic clock from injectable `LongSupplier` / `DoubleSupplier`
- Unexpected disconnect → RECONNECTING up to 3 attempts with delays 0.25/0.5/1.0s (use injectable sleeper)
- `INCOMPATIBLE_DEVICE` → ERROR, no reconnect
- User `disconnect()` → DISCONNECTED, no reconnect
- Malformed packet → increment counter, stay STREAMING

- [ ] **Step 3: Implement `WatchBleWorker` on a dedicated single-thread executor**

Public API:
- `boolean startScan(double timeoutSec)`
- `boolean connect(String deviceId)`
- `void disconnect()`
- `WatchHeartRateSnapshot snapshot(double nowMonotonicSec)`
- `void close()`

Never block the caller on GATT beyond queueing work to the executor.

- [ ] **Step 4: PASS + commit**

```bash
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest --tests com.jagger.rppgbench.watch.WatchBleWorkerTest
git add android/app/src/main/java/com/jagger/rppgbench/watch/WatchBleBackend.java \
  android/app/src/main/java/com/jagger/rppgbench/watch/WatchBleWorker.java \
  android/app/src/test/java/com/jagger/rppgbench/watch/WatchBleWorkerTest.java
git commit -m "feat: add watch BLE worker with bounded reconnect"
```

---

### Task 8: Android BLE backend + permissions

**Files:**
- Create: `android/app/src/main/java/com/jagger/rppgbench/watch/AndroidBleBackend.java`
- Modify: `android/app/src/main/AndroidManifest.xml`
- Modify: packaging test to allow BLE permission strings (still forbid QNN)

- [ ] **Step 1: Update manifest**

Inside `<manifest>`:

```xml
<uses-permission android:name="android.permission.BLUETOOTH_SCAN"
    android:usesPermissionFlags="neverForLocation" />
<uses-permission android:name="android.permission.BLUETOOTH_CONNECT" />
<!-- API < 31 legacy -->
<uses-permission android:name="android.permission.BLUETOOTH" android:maxSdkVersion="30" />
<uses-permission android:name="android.permission.BLUETOOTH_ADMIN" android:maxSdkVersion="30" />
<uses-feature android:name="android.hardware.bluetooth_le" android:required="false" />
```

Keep existing CAMERA permission.

- [ ] **Step 2: Implement `AndroidBleBackend`**

- Scan with `BluetoothLeScanner` for up to timeout; filter service UUID `0000180d-0000-1000-8000-00805f9b34fb` or name containing `huawei`/`heart` (case-insensitive)
- Connect with `BluetoothGatt`; verify HR service; enable notifications on `00002a37-0000-1000-8000-00805f9b34fb`
- Write CCCD for notify; deliver raw bytes to callback
- On GATT disconnect callback invoke `onDisconnect`
- Throw with message `INCOMPATIBLE_DEVICE` when service missing

Use application context; do not touch Camera2 threads.

- [ ] **Step 3: Rebuild APK and audit permissions**

```bash
./scripts/build_android.sh
"$ANDROID_SDK_ROOT/build-tools/36.0.0/aapt2" dump permissions \
  android/app/build/outputs/apk/debug/app-debug.apk
```

Expected: CAMERA + BLUETOOTH_SCAN + BLUETOOTH_CONNECT present.

- [ ] **Step 4: Commit**

```bash
git add android/app/src/main/java/com/jagger/rppgbench/watch/AndroidBleBackend.java \
  android/app/src/main/AndroidManifest.xml tests/test_android_packaging.sh
git commit -m "feat: add Android BLE Heart Rate backend and permissions"
```

---

### Task 9: MainActivity watch UI + status merge + CSV export

**Files:**
- Modify: `MainActivity.java`, `strings.xml`
- Create: `WatchCsvExport.java`
- Create: `WatchCsvExportTest.java` (temp dir assertions)

- [ ] **Step 1: Add UI controls**

In `MainActivity.onCreate`, after camera controls:
- Button「扫描心率设备」
- Spinner for discovered devices
- Buttons「连接手表」「断开手表」
- Text labels for broadcast BPM + watch status
- Disclaimer string: experimental reference, not medical diagnosis

Request Bluetooth permissions before scan/connect (separate from camera).

Own a process-scoped `WatchBleWorker` field; **do not** disconnect it in `onStop` (only camera stops there). Disconnect in explicit button and `onDestroy`/`close`.

- [ ] **Step 2: Merge status every 1s poll**

When camera status JSON is shown, also append watch fields, for example:

```text
watch_status=STREAMING
watch_bpm=72
watch_alignment=ALIGNED
watch_reference_bpm=71
watch_abs_error_bpm=1.2
watch_coverage=0.85
```

Compute alignment when traditional (and deep, if available) heart-rate fields update:
- Parse `bpm`, `heart_rate_valid`, and if present window times from status/events; if window times are not in status JSON yet, extend native status JSON with `window_start_sec` / `window_end_sec` for the latest traditional result (preferred) **or** approximate using last 10s ending at `last_timestamp_sec` only as a temporary fallback documented in UI — prefer extending native status.

**Preferred native status additions** (modify `CameraSessionStatus` + `status_json`):
- `window_start_sec`, `window_end_sec` for latest traditional result
- already have deep fields from ORT work

When a new traditional window end appears, call `WatchAligner.align(...)` with `sessionStartMonotonic` captured at camera start (`SystemClock.elapsedRealtimeNanos()/1e9`).

- [ ] **Step 3: CSV export into session directory**

On camera stop (and optionally periodically), write under the session output directory:
- `watch_heart_rate_samples.csv`
- `watch_rppg_alignments.csv`

Columns per spec §5. Include device label `HUAWEI WATCH GT 5 Pro` or connected name; note receive timestamps are host-side.

- [ ] **Step 4: Unit test CSV header/rows; run unit tests + rebuild APK**

```bash
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest
./scripts/build_android.sh
```

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/java/com/jagger/rppgbench/MainActivity.java \
  android/app/src/main/java/com/jagger/rppgbench/watch/WatchCsvExport.java \
  android/app/src/test/java/com/jagger/rppgbench/watch/WatchCsvExportTest.java \
  android/app/src/main/res/values/strings.xml \
  android/app/src/main/cpp/android_camera_session.* \
  android/app/src/main/cpp/android_jni_handle.cpp
git commit -m "feat: surface watch BLE reference HR and session CSV export"
```

---

### Task 10: Docs + final host verification

**Files:**
- Modify: `ANDROID_NEXT_STEPS.md`, `README.md`
- Optional: `docs/superpowers/plans/2026-07-24-android-onnxruntime-cpu.md` mark superseded by this plan

- [ ] **Step 1: Document exact phone commands**

Include in `ANDROID_NEXT_STEPS.md`:

```bash
# build
export JAVA_HOME=...
export ANDROID_SDK_ROOT=...
export RPPG_OPENCV_ANDROID_SDK=...
export RPPG_ONNXRUNTIME_ANDROID=...
./scripts/build_android.sh

# install
adb install -r android/app/build/outputs/apk/debug/app-debug.apk

# import model
./scripts/import_efficientphys_model.sh \
  /Users/wangjie/.config/superpowers/worktrees/rPPG-qnn-cpp/efficientphys-qnn-export/artifacts/model_export/efficientphys_pure/efficientphys_pure.onnx

# launch
adb shell am start -n com.jagger.rppgbench/.MainActivity
```

List remaining device-only gates from the spec §7.2.

- [ ] **Step 2: Full host verification**

```bash
ctest --test-dir build-camera2-release --output-on-failure
android/gradlew --no-daemon --project-dir android :app:testDebugUnitTest
./scripts/build_android.sh
```

Expected: all green; APK has ORT libs; no model artifact inside APK.

- [ ] **Step 3: Commit docs**

```bash
git add ANDROID_NEXT_STEPS.md README.md
git commit -m "docs: document ORT model import and watch BLE phone gates"
```

---

### Task 11: Device bench checklist (manual; do not fake)

- [ ] **Step 1: Connect phone, install APK, import ONNX, launch**
- [ ] **Step 2: Traditional-only camera path works (face + BPM)**
- [ ] **Step 3: Enable EfficientPhys checkbox after model import; deep fields update without dropping capture FPS hard**
- [ ] **Step 4: Enable watch 心率广播; scan ≤15s; connect; broadcast BPM updates**
- [ ] **Step 5: Confirm alignment metrics on valid windows; STALE within 2s after stopping broadcast**
- [ ] **Step 6: Confirm reconnect ≤3 and user disconnect stays disconnected**
- [ ] **Step 7: Pull session CSVs via `adb shell run-as com.jagger.rppgbench ls files/sessions/...`**
- [ ] **Step 8: Record pass/fail evidence in `ANDROID_NEXT_STEPS.md` or a dated note; never claim device success from host tests alone**

No commit required unless docs updated with evidence.

---

## Spec coverage checklist

| Spec requirement | Task |
|------------------|------|
| ORT CPU EfficientPhys, no QNN/fake fallback | 1, 2 |
| External model + SHA pin + adb import | 2 |
| Input/output contract | 1 (native), 2 (vectors/import) |
| BLE 0x180D/0x2A37 manual scan + ≤3 reconnect | 7, 8 |
| Parser / store / align parity with Python | 4, 5, 6 |
| Camera stop ≠ watch disconnect | 9 |
| UI metrics + disclaimer | 9 |
| CSV export | 9 |
| Permissions | 8 |
| Host tests | 1–7, 10 |
| Device gates | 11 |

## Placeholder / consistency review

- UUIDs, SHA-256 values, reconnect delays, stale 2s, history 180s, alignment thresholds match the approved spec.
- No TBD steps; Java package is consistently `com.jagger.rppgbench.watch`.
- Worker API names are stable across Tasks 7–9 (`startScan`, `connect`, `disconnect`, `snapshot`, `close`).
