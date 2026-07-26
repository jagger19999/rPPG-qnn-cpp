# Android ONNX Runtime CPU implementation

## Fixed contract

- Backend identity: `ONNX_RUNTIME_CPU`; no QNN claim and no fake fallback.
- Runtime: official `onnxruntime-android` 1.27.0 AAR, SHA-256
  `077dec5e2d821234c7dc0aba584bec8f999854b546c754cab93a90741c56fbeb`.
- Model: external app-private `files/models/efficientphys_pure.onnx`; never Git/APK.
- Model SHA-256:
  `c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d`.
- Input boundary: 180 RGB `72x72` frames from `DeepWindowBuilder`; native
  preprocessing performs one population mean/std, appends the last frame and
  transposes to float32 `[181,3,72,72]`.
- Output boundary: finite float32 `[180,1]`.

## Runtime flow

Camera2 capture publishes to a latest-only native frame queue. ROI and the
selected GREEN/POS/CHROM algorithm run in the pipeline. The existing
`DeepWorker` receives six-second windows and runs EfficientPhys asynchronously,
so inference cannot block camera acquisition. Status reports traditional and
deep BPM/quality independently, plus deep inference latency and concrete
runtime errors.

## Verification gates

1. Host CTest must pass.
2. The arm64 APK must cross-build and contain `libonnxruntime.so` but no model,
   checkpoint, dataset, QNN library or fake backend.
3. A real EfficientPhys model with the pinned hash must be imported before the
   deep checkbox is enabled.
4. Frozen-vector parity and camera throughput/thermal behavior remain device
   gates until the external model and phone are available.
