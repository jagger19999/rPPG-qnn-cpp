# Android gated-rPPG implementation baseline

- Baseline commit: `220dbe962410550ae7e6fcf2c787a6a11520a7ee`
- Baseline debug APK SHA-256: `4499410d2da88680af4c2ecf5866d4830eda5ff6befd240d806eb788fb5a2352`
- Baseline APK version: `0.3.0-traditional`
- Deep model: not packaged in the repository; the APK validates and loads the
  TSCAN ONNX file from app storage at runtime, so no baseline model SHA-256 was
  available to record from this worktree.
- The worktree already contained local Android camera, ONNX, UI/chart, spectral
  smoothing, layout and drawable changes. They were preserved in place; this
  implementation did not reset or remove them.

## Shared reference inputs

- Python sample `sample_data/github_rppg/marnixnaber_rPPG_video.mp4`:
  SHA-256 `bbb929217bb19e26bb6a509a6228b1a1356f9eeb1f3660ceaa9298a7406bb0a0`
- Python sample `sample_data/synthetic_cabin.mp4`:
  SHA-256 `2f6dae94d22ec85365f29d576a36e86f0aca067023089831800c2759d29a607a`

The Android repository currently contains deterministic synthetic waveform and
quality/gate reference tests. Physical-camera BPM expectations remain a device
acceptance item because no synchronized watch labels are stored in this
worktree.

## Verified implementation artifact

- Version: `0.4.0-gated` (`versionCode 2`)
- Debug APK SHA-256:
  `9741e4292eee2441f1d5f6a7c399b3734c63f3528665097895946313e3c00c2c`
- APK path: `android/app/build/outputs/apk/debug/app-debug.apk`
- The APK intentionally contains no ONNX, pickle, joblib, PyTorch or Router
  model file. TSCAN still uses a separately installed ONNX file. Router remains
  an explicitly labelled heuristic shadow until a non-smoke model is exported
  and passes the planned parity thresholds.
