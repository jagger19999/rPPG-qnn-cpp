# Android ONNX Runtime CPU EfficientPhys Implementation Plan

**Goal:** Run EfficientPhys through ONNX Runtime CPU in the existing Android
Camera2/OpenCV session while GREEN, POS, or CHROM runs concurrently. Camera
capture and traditional processing must not wait for deep inference.

**Non-goals:** This build does not integrate, probe, or claim QNN, QAIRT,
Adreno, NNAPI, or a fake deep backend. A requested backend must either start
exactly or return a concrete error.

## Frozen contracts

- Runtime identity: `onnxruntime_cpu`.
- Official dependency: ONNX Runtime Android 1.27.0 AAR from Maven Central.
- AAR SHA-256:
  `077dec5e2d821234c7dc0aba584bec8f999854b546c754cab93a90741c56fbeb`.
- Dependency is installed outside Git and linked only for `arm64-v8a`.
- Model filename: `efficientphys_pure.onnx`.
- Model SHA-256:
  `c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d`.
- Input: one finite float32 tensor named `frames`, shape
  `[181, 3, 72, 72]`.
- Output: one finite float32 tensor named `pulse`, shape `[180, 1]`.
- Source window: RGB float values with shape `[180, 72, 72, 3]`, standardized
  with one population mean/std over the complete window, transposed to TCHW,
  then appended with the final frame.
- Model stays external to Git and APK. It is imported into app-private storage
  before session start.

## Task 1: Preserve and verify the Android baseline

- Run the complete host suite from a release build.
- Cross-build the current Camera2/OpenCV/traditional APK.
- Audit the uncommitted diff and preserve all existing work.

## Task 2: Add portable EfficientPhys runtime contracts with TDD

- Add host tests first for exact source shape, finite input, preprocessing,
  output shape, finite output, backend identity, timing, and no fallback.
- Introduce an injected EfficientPhys session boundary so portable
  preprocessing/postprocessing is host-testable without loading an Android
  binary.
- Convert the 180-point pulse waveform to the existing finite
  `HeartRateResult` contract.

## Task 3: Add the native Android ONNX Runtime session

- Install the pinned official AAR outside Git.
- Use the AAR's official C/C++ headers and
  `jni/arm64-v8a/libonnxruntime.so`.
- Validate model input/output count, names, dtypes, and fixed shapes when the
  session is created.
- Configure only ONNX Runtime CPU execution. Do not register another execution
  provider.

## Task 4: Generalize the deep pipeline without fallback

- Permit any explicitly supplied `IDeepRuntime` when deep is enabled.
- Keep `DeepWindowBuilder` at 6 seconds / 180 source frames / 72x72.
- Keep `DeepWorker` latest-only and asynchronous.
- Fail if `onnxruntime_cpu` is requested without its exact runtime factory or
  valid model. Never substitute fake or disabled.

## Task 5: Android model import and status

- Add UI selection for `disabled` or `onnxruntime_cpu`.
- Verify the app-private model SHA-256 before native session creation.
- Surface requested/active backend, model readiness, deep BPM, confidence,
  validity, invalid reason, inference latency, and concrete startup/runtime
  errors.
- Continue to expose the independently selected traditional method.

## Task 6: Verification and handoff

- Run focused red/green tests and then the complete host suite.
- Cross-build the `arm64-v8a` APK with OpenCV and ONNX Runtime.
- Audit APK ABI, native dependencies, permissions, JNI exports, and absence of
  model/checkpoint/vector artifacts and QNN/fake claims.
- Document exact dependency setup, APK install, `adb`/`run-as` model import,
  launch, evidence collection, and remaining phone-only gates.

## Current external evidence gap

The manifest is present, but the actual 224,043,138-byte ONNX file and frozen
`.npy` vectors are not present locally. This does not block implementation,
cross-build, or injected-session tests. It blocks exact local ONNX numerical
comparison and the first real phone inference until the model is supplied with
the frozen SHA-256 above.
