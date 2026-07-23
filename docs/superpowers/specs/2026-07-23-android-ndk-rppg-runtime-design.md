# Android NDK rPPG Runtime Design

## Status

- Date: 2026-07-23
- Target branch: `codex/android-ndk-runtime`
- Baseline: `d5eb3ad`
- Repository: independent `rPPG-qnn-cpp`; the Python/Streamlit `rPPG` repository remains untouched
- Decision: replace the planned Yocto-first deployment path with an Android APK whose signal-processing and inference core stays in C++17

## Objective

Deliver an installable Android `arm64-v8a` application for the Qualcomm bench that:

1. acquires timestamped camera frames through Android's supported camera service;
2. runs one selected traditional method (`GREEN`, `POS`, or `CHROM`);
3. builds the existing 180-frame EfficientPhys input without blocking capture;
4. runs real EfficientPhys inference through the bench's Android QAIRT/QNN GPU libraries once the converter and runtime gates pass;
5. exposes explicit status and writes traceable JSONL/CSV results; and
6. never presents fake inference as a physiological result.

Training, fine-tuning, and dataset handling remain in Python and are not part of the Android package.

## Chosen Architecture

Use a minimal Android application shell around the existing native core:

```text
Android Activity
  permissions, lifecycle, start/stop, compact status UI, export
           |
           | JNI control/events only; no full-frame Java copies
           v
librppg_qnn_core.so (C++17)
  AndroidCameraSource -> RGB frame + sensor/monotonic timestamp
           |
           +--> ROI -> GREEN/POS/CHROM -> BVP/BPM
           |
           +--> 180-frame builder -> latest-only DeepWorker
                                      |
                                      v
                              QnnEfficientPhysRuntime
                                      |
                                      v
                         pulse[180] -> FFT BPM/quality
           |
           v
  AndroidResultSink -> app files + logcat + UI snapshot
```

The Android shell exists because normal application camera permission, lifecycle, package storage, and vendor-library visibility are Android application concerns. Kotlin/Java must not implement rPPG math or copy every camera frame through JNI.

The Android NDK's supported CMake toolchain is
`$ANDROID_NDK/build/cmake/android.toolchain.cmake`; the only initial ABI is
`arm64-v8a`. The exact NDK revision and Android minimum API are supplied by the
bench owner and recorded in every build rather than guessed in source.

Official references:

- Android NDK CMake:
  https://developer.android.com/ndk/guides/cmake
- Android ABI contract:
  https://developer.android.com/ndk/guides/abis
- Camera2 NDK API:
  https://developer.android.com/ndk/reference/group/camera
- Vendor native library declaration:
  https://developer.android.com/guide/topics/manifest/uses-native-library-element
- Android linker namespaces:
  https://source.android.com/docs/core/architecture/vndk/linker-namespace
- Android SELinux:
  https://source.android.com/docs/security/features/selinux

## Deployment Identity Gate

Before camera or QNN integration is declared complete, the bench owner must
identify one supported deployment identity:

1. **Normal APK** (default): camera is exposed by Camera2; QNN libraries are
   packaged legally with the APK or explicitly exposed as vendor native
   libraries.
2. **Privileged/system APK**: required only when the selected camera is marked
   as a system camera or vendor policy restricts it.
3. **Vendor native service**: separate later design if the camera exists only
   as `/dev/video*` and cannot be exposed by Camera2. This is not silently
   implemented inside the normal APK because it requires image integration and
   SELinux policy owned by the device vendor.

The first implementation targets option 1. Failure to enumerate the required
camera is a recorded platform gate, not a reason to bypass Android security or
fall back to an unreported input.

## Reuse Boundary

### Reused without platform behavior changes

- contracts and stable error identifiers;
- `TraditionalPredictor` and GREEN/POS/CHROM math;
- `DeepWindowBuilder`, `DeepWorker`, and latest-only queue;
- EfficientPhys preprocessing and fixed tensor contract;
- waveform-to-BPM postprocessing contract;
- portable result structures;
- host unit tests for pure C++ behavior;
- external checkpoint/ONNX provenance and hashes.

### Adapted behind existing interfaces

- `FrameSource`: add Android Camera2 implementation; retain Linux V4L2 and
  file replay in non-Android builds.
- `IResultSink`: add Android app-storage/logcat implementation; retain host
  JSONL/CSV sink.
- ROI resource discovery: load the cascade from an app asset copied to a
  private regular file, or accept an app-private path.
- configuration: construct `AppConfig` from validated JNI input while retaining
  the existing command-line parser for host/Linux.
- QNN preflight: report Android ABI, API level, resolved backend/model status,
  and linker errors without assuming arbitrary absolute library paths are
  loadable.

### New Android-only components

- Gradle application module and manifest;
- minimal Activity/controller and runtime permission flow;
- JNI bridge with explicit start, stop, status snapshot, and export calls;
- Camera2 NDK session and `AImageReader` frame source;
- YUV-to-RGB conversion with timestamp and stride tests;
- Android-specific storage and log sink;
- Android QNN library/model discovery;
- real `QnnEfficientPhysRuntime`;
- device integration and instrumentation tests.

### Retained but no longer primary

The Yocto toolchain, V4L2 source, Linux executable, and four-file package remain
buildable as a reference target. Android work must not delete them or change
their safe default `--deep disabled`.

## Camera Contract

The application requests a Camera2 output suitable for continuous analysis and
uses native callbacks. The source must:

- enumerate camera IDs and expose the selected ID in status;
- require application camera permission;
- prefer a supported 30 FPS range and record requested versus measured FPS;
- acquire `YUV_420_888` through `AImageReader`;
- correctly handle row and pixel strides;
- convert to the core's explicit RGB/BGR convention exactly once;
- attach a monotonic or sensor timestamp to every accepted frame;
- drop stale images instead of building an unbounded queue;
- report camera disconnect, permission denial, unsupported format, and
  sustained low FPS distinctly;
- stop and release sessions deterministically on Activity stop or explicit
  stop.

Exposure and white-balance locking are optional capabilities. Their availability
and active values must be reported because rPPG is sensitive to automatic color
changes. The first release does not claim fixed exposure/AWB unless the camera
metadata confirms it.

No `/dev/video0` assumption is allowed in the APK path.

## Native/Core Interface

JNI is a control and event boundary:

```text
nativeCreate(config, outputDirectory) -> handle or structured error
nativeStart(handle)
nativeStop(handle)
nativeDestroy(handle)
nativeGetStatus(handle) -> compact immutable snapshot
nativeListCameras() -> camera descriptors or structured error
```

Callbacks to Java carry low-rate state only (heart rate, confidence, FPS,
inference time, status text). Camera image buffers remain native.

All handles are validated, all stop/destroy operations are idempotent, and no
native exception crosses JNI. Native failures map to stable error identifiers
and a Java-visible message.

## QNN and Model Contract

The PyTorch and ONNX reference path remains unchanged:

```text
PURE_EfficientPhys.pth
  -> efficientphys_pure.onnx
  -> Android-target QAIRT converter/context tools
  -> QNN model/context artifact
  -> QnnEfficientPhysRuntime
```

The runtime contract remains:

- source: 180 timestamped RGB ROI frames at 30 FPS;
- normalized input: `float32 [181,3,72,72]` TCHW;
- output: finite `float32 [180,1]` pulse;
- postprocessing: traceable pulse, FFT BPM, confidence/quality, timing;
- no CPU, ONNX Runtime, or fake fallback when QNN mode was requested.

Android QNN libraries must match `arm64-v8a`, the device QAIRT version, and the
application's linker namespace. For target SDK 31 or newer, vendor libraries
that are supplied by the device are declared with `<uses-native-library>` when
the device contract requires it. If the QAIRT package permits app-local
distribution instead, its required shared objects are imported per ABI and
packaged according to the vendor license and sample.

The exact QNN C API calls and library packaging are intentionally gated on the
company's installed QAIRT Android SDK and its examples. Public documentation is
not treated as evidence for a private SDK version.

`--deep fake` remains host-test-only and is never exposed as an Android result
mode.

## OpenCV Strategy

Use the Android `arm64-v8a` OpenCV SDK or a reproducible minimal Android build
containing only the required modules:

- `core`;
- `imgproc`;
- `objdetect`.

Android camera acquisition does not depend on OpenCV `videoio`. The application
imports OpenCV as an ABI-matched prebuilt dependency. Desktop/macOS or Yocto
OpenCV binaries are rejected.

The existing Haar cascade is packaged as an asset for the first vertical
slice. Replacing it with another face detector is a separate accuracy/performance
decision.

## Storage and Observability

Each run receives a unique app-private session directory:

```text
sessions/<session-id>/
  events.jsonl
  heart_rate.csv
  session_summary.json
  pulse_wave.csv
```

Android status is also sent to logcat with a stable tag. Files are published
atomically where the existing sink contract requires it. The UI exports a
completed session through an Android-supported share/export mechanism; the
first bench build may also use `adb` to retrieve app-debuggable files.

No medical claim is added. Results stay marked for research and engineering
validation.

## Threading and Lifecycle

- Camera callback: acquire the newest image, timestamp it, convert it, and
  submit it; never run deep inference.
- Processing thread: ROI and traditional processing.
- Deep worker: one QNN execution at a time with latest-only pending input.
- Output worker: serialize low-rate events and results.
- UI thread: start/stop and render immutable status only.

Stopping the Activity or experiment closes camera submission first, drains or
cancels the deep worker according to the existing close contract, publishes a
summary, then releases QNN and camera resources. A stale result from a prior
session cannot be displayed in a new session.

## Error Handling

Add Android-specific mappings without changing existing numeric meanings:

- camera permission denied;
- requested Camera2 ID unavailable;
- image format/stride unsupported;
- JNI handle or state invalid;
- app output directory unavailable;
- Android QNN library inaccessible through linker namespace;
- QNN model/context missing or incompatible;
- QNN GPU initialization or graph execution failure.

Every error includes the session ID, native backend, and recoverability. The UI
shows the specific reason instead of leaving heart rate blank.

## Delivery Slices

### Slice 1: Android build and portable core

- Gradle/NDK `arm64-v8a` application scaffold;
- root CMake split into portable core and platform entrypoints;
- JNI smoke API and status screen;
- host/Linux build and tests remain green;
- NDK cross-build succeeds without camera/QNN claims.

### Slice 2: Camera2 frame source

- permission and camera enumeration;
- native `AImageReader` acquisition;
- YUV/stride conversion tests;
- recorded timestamps/FPS and lifecycle;
- real-device camera smoke result.

### Slice 3: Traditional live pipeline

- ROI asset loading;
- GREEN/POS/CHROM selection;
- Android result sink and session export;
- controlled camera run with waveform/BPM output.

### Slice 4: QAIRT converter gate

- freeze Android QAIRT/NDK/device versions and hashes;
- convert the existing EfficientPhys ONNX or record the exact unsupported
  operation;
- if conversion fails, approve and test a separate QNN-friendly graph rewrite
  against PyTorch/ONNX reference before changing runtime code.

### Slice 5: Real QNN runtime

- import the exact Android QNN headers/libraries;
- load validated model/context;
- bind fixed tensors and execute on GPU;
- fixed-vector parity against PyTorch and ONNX Runtime;
- no fake or silent fallback.

### Slice 6: Concurrent acceptance

- one selected traditional method plus EfficientPhys;
- camera FPS, deep P50/P95/P99, dropped windows, CPU/GPU/memory/temperature;
- 30-minute stability run;
- controlled seated comparison with synchronized reference timestamps.

## Testing

### Host tests

- all existing Release and sanitizer tests;
- JNI-independent configuration validation;
- YUV plane/stride conversion with deterministic vectors;
- session lifecycle and stale-result isolation;
- Android path selection and error mapping;
- QNN manifest/interface validation.

### Android build tests

- Gradle assemble for `arm64-v8a`;
- APK contains only expected native ABI artifacts;
- no model checkpoint or private dataset is packaged;
- native symbols and shared-library dependencies are audited.

### Device tests

- install and launch;
- permission allow/deny paths;
- camera enumeration and 30 FPS attempt;
- Activity stop/start and camera disconnect recovery;
- POS/CHROM finite output from real frames;
- QNN library visibility and backend identity;
- fixed-vector QNN parity;
- concurrent traditional/deep stability.

Mac/host compilation and an Android APK build do not prove camera, QNN, Adreno,
or physiological success. Those claims require the corresponding device test.

## Acceptance Criteria

The Android migration is complete only when:

1. the existing Linux/native test suite remains green;
2. the `arm64-v8a` APK builds reproducibly from a pinned NDK/Gradle setup;
3. the bench camera is enumerated through the approved Android path and
   produces correctly timestamped frames;
4. POS or CHROM produces finite traceable waveform/BPM without deep inference;
5. the QAIRT converter result and QNN artifacts are versioned by hashes outside
   Git;
6. the real QNN backend reports `QNN GPU`, produces finite `[180,1]` output,
   and passes frozen-vector parity;
7. deep latency cannot reduce capture throughput through back-pressure;
8. normal failures display a concrete reason and never retain an old result;
9. session CSV/JSON/waveform files can be retrieved from the device; and
10. no fake result, training weight, personal frame, or dataset is committed or
    shipped as part of the application.

## Explicit Non-goals

- training or fine-tuning on Android;
- moving Python/Streamlit into the APK;
- simultaneously running multiple deep models;
- medical validation;
- modifying Android vendor SELinux policy without vendor ownership;
- bypassing Camera2 through raw device nodes in a normal application;
- rewriting EfficientPhys before a real converter failure provides evidence.
