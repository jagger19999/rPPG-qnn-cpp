# Android OpenCV Traditional Runtime Implementation Plan

**Goal:** Extend the Camera2 smoke APK so accepted BGR frames run through the
existing Haar ROI and GREEN/POS/CHROM pipeline, write the existing session
artifacts under app-private storage, and expose low-frequency results in the
status UI.

**Boundary:** This slice does not claim QNN, EfficientPhys, or Adreno execution.
Camera enumeration, real image quality, and physiological validity remain
device-only evidence.

## Task 1: Pin and install OpenCV Android

- Use official OpenCV Android SDK 4.13.0.
- Verify SHA-256
  `edfda20fdf65d0bd45391d168ec5261dd30b600b00279c4d910d7f1c3e020f0f`.
- Keep the SDK outside Git and pass its location through
  `RPPG_OPENCV_ANDROID_SDK`.
- Fail the Android build clearly if the SDK or CMake package is absent.

## Task 2: Package the Haar resource

- Package `haarcascade_frontalface_default.xml` as an APK asset.
- Copy it into app-private storage before native processing starts.
- Add packaging checks for the exact asset and forbid external-storage
  permissions.

## Task 3: Feed latest Camera2 frames to the existing pipeline

- Convert each acquired `YUV_420_888` image to packed BGR.
- Submit a cloned `FramePacket` to a latest-only queue.
- Run the existing `Pipeline`, `RoiProcessor`, `TraditionalPredictor`, and
  `ResultSink` on a worker thread.
- Configure `deep=disabled`; do not add fake or QNN execution.
- Stop capture, close the queue, join the worker, and close outputs before
  destroying the native handle.

## Task 4: Expose processing configuration and status

- Add a JNI configuration call for method, cascade path, and output directory.
- Extend status with face state, traditional method, latest BPM/confidence,
  validity, processing exit code, and output directory.
- Keep JNI entry points `noexcept` and stable-error mapped.

## Task 5: Validate without a device

- Keep all host tests green, including synthetic GREEN/POS/CHROM and session
  output tests.
- Cross-build the `arm64-v8a` debug APK with OpenCV.
- Audit that the APK contains only `arm64-v8a`, OpenCV, the project JNI
  library, and the Haar asset; no QNN or deep model artifacts.
- Record camera enumeration and live physiological validation as the remaining
  device gates.
