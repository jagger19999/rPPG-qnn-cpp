# TSCAN Numeric Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Python, ONNX Runtime, and Android/C++ produce equivalent TSCAN inputs, waveforms, BPM, confidence, and validity for the same 180-frame RGB window.

**Architecture:** Extract TSCAN preprocessing and postprocessing from the Android ONNX runtime into host-testable core components. Split deep full-face ROI extraction from the traditional cheek ROI while retaining the existing shared face detector and `DeepWindowBuilder`. A deterministic Python reference-vector generator is the numerical source of truth, and every production-path replacement is preceded by a failing contract test.

**Tech Stack:** Python 3.12, NumPy, PyTorch, ONNX Runtime, C++17, OpenCV 4, CMake/CTest, Android NDK, ONNX Runtime Android.

---

## Scope and repository boundaries

This plan implements only stages 1–5 of the approved design: reference vectors, preprocessing, deep ROI separation, postprocessing, and end-to-end numerical wiring. Timing decomposition, performance optimization, WebView repair, and simulation mode require later plans.

Two repositories participate:

- Python reference: `/Users/wangjie/Documents/keti/rPPG`
- C++/Android implementation: `/Users/wangjie/Documents/keti/rPPG-qnn-cpp-android`

Do not stage the pre-existing uncommitted Android/UI/traditional-algorithm changes unless a task explicitly names the file. In particular, `android_onnx_cpu_runtime.cpp` already contains user changes; inspect its current diff immediately before Task 6 and preserve unrelated hunks.

## File responsibility map

### Python repository

- Create `tests/test_tscan_reference_vector.py`: contract tests for deterministic RGB generation, preprocessing, postprocessing, and manifest hashes.
- Create `scripts/generate_tscan_reference_vector.py`: standalone generator for the `.npz` vector and JSON manifest; imports the existing `deep_live` reference functions rather than duplicating formulas.
- Create `tests/fixtures/tscan_reference_manifest.json`: small committed metadata and hashes. Do not commit the checkpoint or ONNX model.
- Generated, Git-ignored `tests/fixtures/tscan_reference_vector.npz`: deterministic input/preprocessed tensor and optional waveform used while running parity tests.

### C++/Android repository

- Create `include/rppg_qnn/tscan_preprocessor.hpp` and `src/tscan_preprocessor.cpp`: RGB NHWC to TSCAN NCHW conversion only.
- Create `tests/test_tscan_preprocessor.cpp`: exact formula, layout, last-frame, invalid-input, and Python-vector tests.
- Create `include/rppg_qnn/tscan_postprocessor.hpp` and `src/tscan_postprocessor.cpp`: waveform to BPM/confidence conversion only.
- Create `tests/test_tscan_postprocessor.cpp`: Hann-window and in-band FFT parity tests.
- Modify `include/rppg_qnn/roi_processor.hpp` and `src/roi_processor.cpp`: expose a pure expanded-face rectangle/crop operation without changing cheek ROI behavior.
- Modify `tests/test_roi_processor.cpp`: deep ROI geometry and traditional ROI non-regression tests.
- Modify `include/rppg_qnn/deep_window_builder.hpp` and `src/deep_window_builder.cpp`: accept frames already selected for the deep path; retain sampling behavior.
- Modify `android/app/src/main/cpp/android_camera_session.cpp`: feed expanded-face images to the deep builder while traditional predictors continue receiving cheek ROI.
- Modify `android/app/src/main/cpp/android_onnx_cpu_runtime.cpp`: compose preprocessor, ONNX session, and postprocessor.
- Modify `CMakeLists.txt` and `android/app/src/main/cpp/CMakeLists.txt`: compile new core components and tests.

## Task 1: Create the deterministic Python reference contract

**Files:**
- Create: `/Users/wangjie/Documents/keti/rPPG/tests/test_tscan_reference_vector.py`
- Create: `/Users/wangjie/Documents/keti/rPPG/scripts/generate_tscan_reference_vector.py`
- Create: `/Users/wangjie/Documents/keti/rPPG/tests/fixtures/tscan_reference_manifest.json`
- Modify: `/Users/wangjie/Documents/keti/rPPG/.gitignore`

- [ ] **Step 1: Write the failing deterministic-vector test**

Add a test that imports `build_reference_frames` and `build_reference_payload` from the not-yet-created generator:

```python
def test_reference_payload_matches_live_tscan_contract():
    frames = build_reference_frames()
    payload = build_reference_payload(frames)
    assert frames.shape == (180, 72, 72, 3)
    assert frames.dtype == np.float32
    assert payload["model_input"].shape == (180, 6, 72, 72)
    assert payload["model_input"].dtype == np.float32
    assert np.all(payload["model_input"][-1, :3] == 0.0)
    expected = np.concatenate(
        [_diff_normalize(frames), _standardize(frames)], axis=-1
    ).transpose(0, 3, 1, 2)
    np.testing.assert_allclose(payload["model_input"], expected, atol=0.0, rtol=0.0)
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cd /Users/wangjie/Documents/keti/rPPG
.venv/bin/pytest tests/test_tscan_reference_vector.py -q
```

Expected: collection fails with `ModuleNotFoundError` for `scripts.generate_tscan_reference_vector`.

- [ ] **Step 3: Implement deterministic RGB and preprocessing payload generation**

Use analytic spatial and temporal terms so the file is reproducible without camera or codec dependencies:

```python
def build_reference_frames() -> np.ndarray:
    t = np.arange(180, dtype=np.float32)[:, None, None, None]
    y = np.arange(72, dtype=np.float32)[None, :, None, None]
    x = np.arange(72, dtype=np.float32)[None, None, :, None]
    channel = np.arange(3, dtype=np.float32)[None, None, None, :]
    pulse = 2.5 * np.sin(2.0 * np.pi * 1.2 * t / 30.0)
    frames = 80.0 + 0.31 * x + 0.17 * y + 11.0 * channel + pulse * (channel + 1.0)
    return np.asarray(frames, dtype=np.float32)


def build_reference_payload(frames: np.ndarray) -> dict[str, np.ndarray]:
    model_input = np.concatenate(
        [_diff_normalize(frames), _standardize(frames)], axis=-1
    ).transpose(0, 3, 1, 2)
    return {"frames": frames, "model_input": np.ascontiguousarray(model_input)}
```

The generator must accept `--output`, `--manifest`, and optional `--onnx`. When `--onnx` is supplied, run the model and add `waveform`, `bpm`, and `confidence`; otherwise generate only frames and model input. Write SHA-256 for each array's raw C-order bytes into the JSON manifest.

- [ ] **Step 4: Run the Python contract test and generator**

Run:

```bash
cd /Users/wangjie/Documents/keti/rPPG
.venv/bin/pytest tests/test_tscan_reference_vector.py -q
.venv/bin/python scripts/generate_tscan_reference_vector.py \
  --output tests/fixtures/tscan_reference_vector.npz \
  --manifest tests/fixtures/tscan_reference_manifest.json \
  --onnx ubfc_tscan_full_lr3e-5_Epoch10.onnx
```

Expected: tests pass; manifest reports shapes `[180,72,72,3]`, `[180,6,72,72]`, `[180,1]`; all arrays are finite.

- [ ] **Step 5: Ignore the generated binary and commit the reproducible contract**

Append exactly:

```gitignore
tests/fixtures/tscan_reference_vector.npz
```

Then run:

```bash
git add .gitignore scripts/generate_tscan_reference_vector.py \
  tests/test_tscan_reference_vector.py tests/fixtures/tscan_reference_manifest.json
git commit -m "test: freeze TSCAN Python reference contract"
```

Expected: commit includes the generator, tests, manifest, and ignore rule but no model or `.npz` file.

## Task 2: Extract and align the C++ TSCAN preprocessor

**Files:**
- Create: `include/rppg_qnn/tscan_preprocessor.hpp`
- Create: `src/tscan_preprocessor.cpp`
- Create: `tests/test_tscan_preprocessor.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Add a failing C++ formula and layout test**

Define the intended API in the test:

```cpp
const rppg_qnn::TscanTensor output =
    rppg_qnn::preprocess_tscan_rgb(input);
EXPECT_EQ(output.shape,
          (std::vector<std::int64_t>{180, 6, 72, 72}));
EXPECT_EQ(output.values.size(), 180U * 6U * 72U * 72U);
for (std::size_t channel = 0; channel < 3U; ++channel) {
  const std::size_t last = (179U * 6U + channel) * 72U * 72U;
  for (std::size_t pixel = 0; pixel < 72U * 72U; ++pixel) {
    EXPECT_EQ(output.values[last + pixel], 0.0F);
  }
}
```

Create a small analytic input where one pixel can be calculated manually and assert:

```cpp
expected_diff = ((next - current) / (next + current + 1e-7F)) / diff_std;
```

Also assert RGB channel order and NCHW offsets.

- [ ] **Step 2: Add the target and verify RED**

Register `test_tscan_preprocessor` in `CMakeLists.txt`, then run in a fresh build directory because existing build caches contain obsolete worktree paths:

```bash
cmake -S . -B build-tscan-parity -DBUILD_TESTING=ON
cmake --build build-tscan-parity --target test_tscan_preprocessor -j4
```

Expected: compile fails because `rppg_qnn/tscan_preprocessor.hpp` does not exist.

- [ ] **Step 3: Implement the minimal public API**

Create:

```cpp
struct TscanTensor {
  std::vector<float> values;
  std::vector<std::int64_t> shape;
};

TscanTensor preprocess_tscan_rgb(const DeepInput& input);
```

Implementation requirements:

1. Require finite `[180,72,72,3]` RGB input.
2. Compute 179 normalized frame differences using the approved ratio formula.
3. Compute population standard deviation over those 179 diff frames.
4. Store normalized differences in output frames 0–178 and zeros in frame 179.
5. Standardize appearance over all 180 RGB frames using population mean/std.
6. Pack `[diff_R,diff_G,diff_B,appearance_R,appearance_G,appearance_B]` into NCHW.
7. Throw `AppError(ErrorCode::InferenceFailed, "TSCAN_PREPROCESS_...")` for invalid shape, nonfinite values, or zero variance.

- [ ] **Step 4: Verify GREEN and the existing deep-window tests**

Run:

```bash
cmake --build build-tscan-parity --target test_tscan_preprocessor test_deep_window_builder -j4
ctest --test-dir build-tscan-parity -R 'tscan_preprocessor|deep_window_builder' --output-on-failure
```

Expected: both tests pass.

- [ ] **Step 5: Add Python-vector parity without committing the `.npz`**

Extend the Python generator with `--cpp-header` to emit a compact set of selected reference values and hashes rather than the full tensor. The C++ test must compare at least frames `{0,1,89,178,179}`, all six channels, and pixels `{0,73,2591,5183}` with absolute tolerance `1e-5`.

Run the generator, rebuild, and expect `test_tscan_preprocessor` to pass.

- [ ] **Step 6: Commit the preprocessor slice**

```bash
git add CMakeLists.txt include/rppg_qnn/tscan_preprocessor.hpp \
  src/tscan_preprocessor.cpp tests/test_tscan_preprocessor.cpp
git commit -m "fix: align C++ TSCAN preprocessing"
```

## Task 3: Extract and align TSCAN postprocessing

**Files:**
- Create: `include/rppg_qnn/tscan_postprocessor.hpp`
- Create: `src/tscan_postprocessor.cpp`
- Create: `tests/test_tscan_postprocessor.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Write the failing 72 BPM reference test**

Use a 1.2 Hz sinusoid sampled at 30 Hz for 180 samples:

```cpp
std::vector<float> waveform(180U);
for (std::size_t i = 0; i < waveform.size(); ++i) {
  waveform[i] = static_cast<float>(3.0 + std::sin(2.0 * kPi * 1.2 * i / 30.0));
}
const auto result = rppg_qnn::postprocess_tscan_waveform(waveform, 0.10);
EXPECT_TRUE(result.is_valid);
EXPECT_EQ(result.bpm, 70.0);
EXPECT_TRUE(result.confidence > 0.0 && result.confidence <= 1.0);
```

The expected discrete FFT bin is 70 BPM, not the continuous 72 BPM frequency, because 180 samples at 30 Hz have 10 BPM resolution.

- [ ] **Step 2: Add invalid waveform tests**

Assert concrete reasons for empty, nonfinite, constant, and low-confidence inputs:

```cpp
EXPECT_EQ(result.invalid_reason, "TSCAN_WAVEFORM_INVALID");
EXPECT_EQ(result.invalid_reason, "TSCAN_LOW_CONFIDENCE");
```

- [ ] **Step 3: Register the target and verify RED**

Run:

```bash
cmake --build build-tscan-parity --target test_tscan_postprocessor -j4
```

Expected: compile fails because the postprocessor header/API is missing.

- [ ] **Step 4: Implement the approved postprocessor contract**

Create:

```cpp
struct TscanPostprocessResult {
  double bpm{0.0};
  double confidence{0.0};
  bool is_valid{false};
  std::string invalid_reason;
};

TscanPostprocessResult postprocess_tscan_waveform(
    const std::vector<float>& waveform,
    double confidence_threshold = 0.10);
```

Implementation must subtract the mean, apply `0.5 - 0.5*cos(2*pi*n/(N-1))`, calculate the DFT bins, restrict both peak and total-power calculations to `0.75–2.5 Hz`, and calculate confidence as peak/in-band total.

- [ ] **Step 5: Verify against Python values and commit**

Add Python-produced expected BPM/confidence for three deterministic waveforms, then run:

```bash
cmake --build build-tscan-parity --target test_tscan_postprocessor -j4
ctest --test-dir build-tscan-parity -R tscan_postprocessor --output-on-failure
git add CMakeLists.txt include/rppg_qnn/tscan_postprocessor.hpp \
  src/tscan_postprocessor.cpp tests/test_tscan_postprocessor.cpp
git commit -m "fix: align TSCAN heart-rate postprocessing"
```

Expected: tests pass and confidence absolute error is below `1e-4`.

## Task 4: Separate deep full-face ROI from traditional cheek ROI

**Files:**
- Modify: `include/rppg_qnn/roi_processor.hpp`
- Modify: `src/roi_processor.cpp`
- Modify: `tests/test_roi_processor.cpp`
- Modify: `include/rppg_qnn/contracts.hpp`

- [ ] **Step 1: Write failing pure geometry tests**

Add tests for a centered face, a face near each image boundary, and rounding behavior:

```cpp
const FaceBox face{100, 80, 40, 60, 1.0};
const auto rect = rppg_qnn::expanded_face_roi(face, {320, 240}, 1.5);
EXPECT_EQ(*rect, cv::Rect(90, 65, 60, 90));
```

Assert existing `cheek_roi(face, size)` still returns its previous rectangle.

- [ ] **Step 2: Verify RED**

Run:

```bash
cmake --build build-tscan-parity --target test_roi_processor -j4
```

Expected: compile fails because `expanded_face_roi` is missing.

- [ ] **Step 3: Implement expanded-face geometry and carry both ROIs**

Add:

```cpp
std::optional<cv::Rect> expanded_face_roi(
    const FaceBox& face, cv::Size frame_size, double scale = 1.5);
```

Use the same center/round/clip order as Python `_expanded_face_crop`. Extend `RoiPacket` with a separate `cv::Mat deep_roi_bgr`; keep existing `roi_bgr` as the traditional cheek ROI to minimize compatibility risk.

- [ ] **Step 4: Update all `RoiPacket` aggregate initializers**

Compile errors will identify every initializer. Supply `{}` for `deep_roi_bgr` in tests that exercise only traditional behavior. In `RoiProcessor::process`, populate both cloned images from the same frame and face box.

- [ ] **Step 5: Run ROI, traditional, window, and pipeline regression tests**

```bash
cmake --build build-tscan-parity --target \
  test_roi_processor test_traditional_predictor test_deep_window_builder test_pipeline -j4
ctest --test-dir build-tscan-parity \
  -R 'roi_processor|traditional_predictor|deep_window_builder|pipeline' \
  --output-on-failure
```

Expected: all pass; traditional ROI assertions remain unchanged.

- [ ] **Step 6: Commit the ROI boundary slice**

```bash
git add include/rppg_qnn/roi_processor.hpp include/rppg_qnn/contracts.hpp \
  src/roi_processor.cpp tests/test_roi_processor.cpp \
  tests/test_deep_window_builder.cpp tests/test_pipeline.cpp
git commit -m "fix: separate TSCAN full-face ROI"
```

## Task 5: Feed only the deep ROI into `DeepWindowBuilder`

**Files:**
- Modify: `src/pipeline.cpp`
- Modify: `tests/test_pipeline.cpp`
- Modify: `tests/test_deep_window_builder.cpp`

- [ ] **Step 1: Write a failing pipeline routing test**

Create a packet whose traditional ROI is solid BGR `(1,2,3)` and deep ROI is solid BGR `(20,40,80)`. Capture the first submitted `DeepInput` and assert its RGB values are `(80,40,20)`, proving the builder received `deep_roi_bgr`.

- [ ] **Step 2: Run and verify RED**

```bash
cmake --build build-tscan-parity --target test_pipeline -j4
ctest --test-dir build-tscan-parity -R pipeline --output-on-failure
```

Expected: assertion shows the old cheek ROI values.

- [ ] **Step 3: Route a deep-specific packet into the builder**

At the deep builder call site construct a packet retaining frame metadata and face metadata while moving/cloning `deep_roi_bgr` into the field consumed by `DeepWindowBuilder`. Do not change the packet sent to traditional predictors.

- [ ] **Step 4: Run regression tests and commit**

```bash
cmake --build build-tscan-parity --target \
  test_pipeline test_deep_window_builder test_traditional_predictor -j4
ctest --test-dir build-tscan-parity \
  -R 'pipeline|deep_window_builder|traditional_predictor' --output-on-failure
git add src/pipeline.cpp tests/test_pipeline.cpp tests/test_deep_window_builder.cpp
git commit -m "fix: route full-face frames to deep windows"
```

## Task 6: Recompose the Android ONNX runtime from tested components

**Files:**
- Modify: `android/app/src/main/cpp/android_onnx_cpu_runtime.cpp`
- Modify: `android/app/src/main/cpp/CMakeLists.txt`
- Create or modify: `tests/test_tscan_runtime.cpp`
- Modify: `CMakeLists.txt`

- [ ] **Step 1: Inspect and preserve the user's current runtime diff**

Run:

```bash
git diff -- android/app/src/main/cpp/android_onnx_cpu_runtime.cpp
```

Record which hunks belong to the current TSCAN migration. Do not overwrite model naming, session options, input/output names, or unrelated user work.

- [ ] **Step 2: Write a host-side composition test with a recording session**

Define a session abstraction accepting a ready `[180,6,72,72]` tensor and returning a known waveform. Assert the composed runtime:

- calls preprocessing once;
- sends the exact TSCAN shape;
- returns postprocessor BPM/confidence;
- preserves source-window metadata;
- returns stable errors for preprocessing and model-output failures.

- [ ] **Step 3: Verify RED**

Run:

```bash
cmake --build build-tscan-parity --target test_tscan_runtime -j4
```

Expected: compile fails until the runtime/session boundary exists.

- [ ] **Step 4: Replace duplicated runtime algorithms with composition**

In `android_onnx_cpu_runtime.cpp`, replace its local `preprocess` and DFT/BPM implementation with:

```cpp
const TscanTensor model_input = preprocess_tscan_rgb(input);
const std::vector<float> waveform = run_onnx(model_input);
const TscanPostprocessResult hr = postprocess_tscan_waveform(waveform);
```

The Android CMake target must compile/link `src/tscan_preprocessor.cpp` and `src/tscan_postprocessor.cpp`. Keep ONNX execution and Android-specific error translation in the Android runtime.

- [ ] **Step 5: Build host tests and Android debug APK**

Run:

```bash
cmake --build build-tscan-parity --target \
  test_tscan_runtime test_tscan_preprocessor test_tscan_postprocessor -j4
ctest --test-dir build-tscan-parity \
  -R 'tscan_runtime|tscan_preprocessor|tscan_postprocessor' --output-on-failure
cd android
./gradlew :app:assembleDebug
```

Expected: all selected tests pass and `app-debug.apk` builds successfully.

- [ ] **Step 6: Commit only the runtime composition files**

```bash
git add android/app/src/main/cpp/android_onnx_cpu_runtime.cpp \
  android/app/src/main/cpp/CMakeLists.txt CMakeLists.txt \
  tests/test_tscan_runtime.cpp
git commit -m "fix: compose Android TSCAN runtime from parity components"
```

Before committing, inspect `git diff --cached` and confirm no unrelated existing Android/UI changes are staged.

## Task 7: Run the numerical-parity completion gate

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-07-26-tscan-android-parity-and-runtime-design.md` only if actual verified thresholds differ from the approved thresholds; do not relax thresholds to make tests pass.

- [ ] **Step 1: Regenerate the Python reference using the exact deployed ONNX file**

```bash
cd /Users/wangjie/Documents/keti/rPPG
.venv/bin/python scripts/generate_tscan_reference_vector.py \
  --output tests/fixtures/tscan_reference_vector.npz \
  --manifest tests/fixtures/tscan_reference_manifest.json \
  --onnx ubfc_tscan_full_lr3e-5_Epoch10.onnx
.venv/bin/pytest tests/test_tscan_reference_vector.py -q
```

Expected: manifest hashes match and Python tests pass.

- [ ] **Step 2: Reconfigure and run the complete host suite**

```bash
cd /Users/wangjie/Documents/keti/rPPG-qnn-cpp-android
cmake -S . -B build-tscan-parity -DBUILD_TESTING=ON
cmake --build build-tscan-parity -j4
ctest --test-dir build-tscan-parity --output-on-failure
```

Expected: all configured tests pass. Failures caused by missing optional external SDK/model prerequisites must be explicitly identified; do not report the suite as passing if tests were skipped or not run.

- [ ] **Step 3: Record measured numerical evidence**

Document in `README.md`:

- model SHA-256;
- reference-vector manifest SHA-256;
- maximum preprocessing absolute error;
- PyTorch/ONNX maximum waveform error;
- Python/C++ Pearson correlation;
- BPM and confidence difference;
- host and Android verification status.

Do not claim Android physiological accuracy or performance without target-device evidence.

- [ ] **Step 4: Build the APK one final time**

```bash
cd android
./gradlew :app:assembleDebug
```

Expected: `BUILD SUCCESSFUL` and the APK contains native libraries for `arm64-v8a`.

- [ ] **Step 5: Commit the verified parity report**

```bash
git add README.md
git commit -m "docs: record verified TSCAN numerical parity"
```

## Completion checklist

- [ ] Python reference generation is deterministic and hash-checked.
- [ ] C++ preprocessing matches Python within `1e-5` maximum absolute error.
- [ ] TSCAN receives expanded full-face ROI while traditional methods retain cheek ROI.
- [ ] C++ postprocessing matches Python within `1e-4` confidence error.
- [ ] Android runtime contains no duplicate preprocessing or heart-rate DFT logic.
- [ ] Fixed-input waveform correlation exceeds `0.9999`.
- [ ] BPM is identical for the fixed vector or differs by no more than one FFT bin.
- [ ] Fresh CMake build and selected Android Gradle build succeed.
- [ ] Existing user changes remain preserved and unrelated files remain unstaged.
