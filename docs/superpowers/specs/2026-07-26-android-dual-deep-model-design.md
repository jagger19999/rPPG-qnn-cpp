# Android Selectable TSCAN and EfficientPhys Design

## Goal

Make the Android deep pipeline selectable in the same way as the traditional
pipeline: `关闭 / TSCAN / EfficientPhys`. Exactly one deep model runs at a time,
alongside the selected traditional method. Add stage-level timing and explicit
signal-quality handling so slow or inaccurate deep results can be diagnosed
instead of being silently presented as trustworthy.

## Verified model assets

The models remain external to Git and the APK, but both already exist in app
private storage on the connected phone.

| Model | App-private filename | SHA-256 | Input | Output |
|---|---|---|---|---|
| TSCAN | `ubfc_tscan_full_lr3e-5_Epoch10.onnx` | `342a3c8033dda9ab154e85d5a4e2a876a6461648b7fcb27c46a7023e662bcc64` | `[180,6,72,72]` | `[180,1]` |
| EfficientPhys | `efficientphys_pure.onnx` | `c1b321042db1335da70b0295cc84f653a2cfe90f75cff738b3045ea3c103257d` | `[181,3,72,72]` | `[180,1]` |

TSCAN uses a UBFC-trained checkpoint. EfficientPhys uses the official PURE
checkpoint. Their dataset difference is always exposed in diagnostics because
it affects cross-camera generalization and makes a direct accuracy guarantee
invalid.

## User interaction

Replace the deep checkbox with a spinner labelled `深度模型` containing:

- `关闭`
- `TSCAN (UBFC)`
- `EfficientPhys (PURE)`

Like the traditional method spinner, the selection is made before capture.
Both method selectors are disabled during capture and re-enabled after stop.
Changing either selection starts a new result history on the next capture; it
never hot-swaps a runtime inside an active session.

The deep BPM card and waveform card use the selected model name. Disabled,
sampling, inference, invalid, and valid states remain explicit. The UI never
labels a low-confidence result as trustworthy merely because an FFT peak was
found.

## Runtime architecture

`TraditionalProcessingConfig` replaces its `deep_enabled` boolean with a
`DeepModel` enum (`Disabled`, `Tscan`, `EfficientPhys`) plus the selected model
path. The pipeline continues to own one latest-only `DeepWorker`, so camera
capture is not blocked and obsolete windows are discarded.

Runtime construction is model-specific:

- TSCAN: reuse the verified 180-frame uniform RGB window, TSCAN difference plus
  appearance preprocessing, `[180,6,72,72]` ORT session, and TSCAN postprocessor.
- EfficientPhys: reuse the same uniform 180-frame RGB source window, perform one
  global population mean/std normalization over the full window, transpose to
  TCHW, append a duplicate of the final normalized frame, run the
  `[181,3,72,72]` ORT session, and apply the common waveform postprocessor.

The ORT session inspects model input/output type, names, and shapes at load time.
It rejects a TSCAN file selected as EfficientPhys or the reverse. Each expected
filename is also checked against the pinned SHA-256 before the camera starts.
There is no silent fallback to the other model, fake inference, QNN, or a
traditional method.

## Performance instrumentation

Each deep result records these independent durations:

- `window_materialization_ms`: resize, BGR-to-RGB conversion, and construction
  of the uniform 180-frame source tensor
- `preprocess_ms`: model-specific normalization/layout conversion
- `runtime_ms`: only `Ort::Session::Run`
- `postprocess_ms`: waveform validation, FFT, confidence, and stability logic
- `inference_ms`: end-to-end preprocess + runtime + postprocess, retained for
  compatibility

Status JSON and the deep card expose the latest stage timings. Session CSV keeps
the same fields so TSCAN and EfficientPhys can be compared after a run.

The implementation reuses preallocated preprocessing/output buffers where
ownership is local to one deep worker. It does not reduce the 72×72 resolution,
the 180-sample window, or model precision before parity and accuracy baselines
exist. ORT thread counts are benchmarked at 2, 4, and 6 on the connected phone;
the selected default is the fastest sustained configuration that does not cause
a material camera FPS regression. Thread tuning is model-specific but is not a
user-facing control.

The current EfficientPhys ONNX contains about 215 MB of constants produced by
the exported temporal-shift graph. This phase runs it faithfully first. ONNX
graph rewriting, FP16/INT8 conversion, and QNN/NNAPI acceleration are separate
optimization phases and require their own numerical parity gates.

## Accuracy and stability

The app preserves two values:

- `raw_bpm`: the spectral result for the current waveform
- `display_bpm`: a transparent temporal stabilization of valid results from the
  same selected deep model

Stabilization never substitutes a traditional or watch BPM. It uses only deep
model evidence:

1. Reject non-finite/constant waveform and confidence below the calibrated
   threshold.
2. Evaluate fundamental, half-frequency, and double-frequency candidates from
   the same spectrum; accept a harmonic correction only when the fundamental
   support ratio passes a documented threshold.
3. Reject a large one-window jump when it is unsupported by the spectrum;
   otherwise allow it and reset temporal history.
4. Use the median of the last three accepted deep BPM values as `display_bpm`.

Raw BPM, stabilized BPM, confidence, correction reason, and validity are all
exported. Traditional/deep and watch/deep disagreement produces a warning only;
it never changes either deep value. This keeps evaluation honest.

Thresholds are not guessed from one live session. First retain the existing
thresholds, collect paired windows from the phone and watch, then calculate MAE,
invalid rate, harmonic-error rate, and latency for both models. A later tuning
commit may change thresholds with those results as evidence.

## Waveforms and results

The existing deep waveform card becomes model-neutral. It displays the selected
model's latest 180-point waveform and fixed `-6.0 s → 0 s` time axis. Switching
models between sessions clears the deep waveform and stabilization history while
leaving traditional processing independent.

The session export identifies `model`, checkpoint SHA-256, dataset label, raw
BPM, displayed BPM, all timing fields, confidence, validity, and correction
reason. This prevents results from different weights being mixed during later
accuracy analysis.

## Failure behavior

- Missing/hash-mismatched selected model: refuse deep start with a concrete
  model error; traditional capture remains available after selecting `关闭`.
- Shape/name/type mismatch: refuse that model before accepting camera frames.
- Allocation or ORT failure: publish an invalid deep result and keep traditional
  and camera paths alive.
- Deep inference slower than submission: latest-only queue replaces obsolete
  windows and reports replacement count.
- Low source FPS or capture gap: do not infer on a malformed time window.

## Testing and device gates

Host tests cover model selection parsing, filename/hash policy, both exact
preprocessing contracts, model-shape rejection, timing fields, latest-only
behavior, harmonic/jump decisions, three-value median, and history reset on model
change.

Android unit tests cover spinner-to-config mapping, locked controls while
running, model-specific labels/states, and metadata parsing. Packaging tests
continue to prohibit model files under `android/`.

Connected-device gates:

1. TSCAN and EfficientPhys files match pinned hashes in app-private storage.
2. `关闭`, `TSCAN`, and `EfficientPhys` each start with the expected backend.
3. Both models produce a BPM and 180-point waveform without changing portrait
   camera orientation.
4. Stage timings and replacement counts are visible and exported.
5. Sustained camera FPS and memory are recorded for both models.
6. A paired watch session produces per-model MAE and invalid-rate reports; no
   claim that one model is more accurate is made without this evidence.
