# EfficientPhys ONNX Reference Package Design

## Objective

Build the host-side reference package for the public rPPG-Toolbox
`PURE_EfficientPhys.pth` checkpoint without training or modifying the original
Python rPPG repository. The package must prove that the official PyTorch model
can be loaded strictly, exported as a fixed-shape ONNX graph, and reproduced by
ONNX Runtime on the Mac before any QAIRT/QNN conversion is attempted.

This phase is an offline model-engineering step for the independent
`rPPG-qnn-cpp` repository. It does not add Python to the Linux bench package,
does not implement `libQnnGpu.so` inference, and does not claim physiological
accuracy.

## Confirmed assumptions

1. The source checkpoint is external to Git at
   `/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth`.
2. Its SHA256 is
   `e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49`.
3. The checkpoint is an `OrderedDict` containing 21 tensors, and every key has
   one `module.` prefix. Removing that prefix allows a strict load with no
   missing or unexpected keys.
4. The official inference configuration uses 30 FPS, RGB, 72 x 72 frames,
   `Standardized` input, a 180-frame chunk, and `FRAME_DEPTH=10`.
5. The trainer flattens the 180-frame chunk, then duplicates its final frame.
   The model therefore receives `[181, 3, 72, 72]`; its internal
   `torch.diff(dim=0)` produces 180 frames and the output is `[180, 1]`.
6. Feeding only 180 frames is invalid because 179 post-difference frames are
   not divisible by `FRAME_DEPTH=10`.
7. No QAIRT SDK or target `libQnnGpu.so`, `libQnnSystem.so`, or
   `libOpenCL.so` is currently available on this Mac.

## Approaches considered

### A. Fixed-shape export of the official graph — selected

Load the official implementation and checkpoint, build the exact 181-frame
tensor expected by the trainer, and export a static ONNX graph. This is the
smallest change and gives the strongest provenance for later QNN comparison.

### B. Rewrite TSM and differencing into a QNN-friendly graph now — deferred

Replacing tensor writes, `torch.diff`, or the TSM implementation may make a
future QNN converter happier, but it introduces a second model before the
official graph has a frozen reference. It may be added only after a real QAIRT
converter rejects the reference graph, and only with layer/output parity tests.

### C. Reimplement EfficientPhys directly in C++ — rejected for this phase

This would combine model translation, preprocessing, QNN integration, and
numerical debugging. It is slower to validate and makes discrepancies hard to
localize.

## Model and preprocessing contract

The source window contains exactly 180 RGB ROI frames at 30 FPS with shape
`[180, 72, 72, 3]` and `uint8` values.

Preprocessing must reproduce the rPPG-Toolbox behavior:

1. Convert the complete 180-frame window to floating point.
2. Compute one scalar mean and one scalar population standard deviation across
   all frames, pixels, and channels.
3. Apply `(data - mean) / std` globally. Non-finite results, including the
   zero-variance case, become zero.
4. Convert to `float32` and transpose RGB `THWC` to `TCHW`.
5. Append one duplicate of frame 179, producing `[181, 3, 72, 72]`.

The stable ONNX interface is:

| Field | Contract |
|---|---|
| Input name | `frames` |
| Input dtype/layout | `float32`, fixed `TCHW` |
| Input shape | `[181, 3, 72, 72]` |
| Output name | `pulse` |
| Output dtype | `float32` |
| Output shape | `[180, 1]` |
| Frame rate | 30 FPS |
| Frame depth | 10 |
| Color order | RGB |

No dynamic time, batch, height, or width dimensions are allowed in this first
reference graph.

## Architecture and data flow

```text
external checkpoint + official model source
                  |
                  v
       strict checkpoint loader
                  |
180 RGB frames -> standardize -> TCHW -> duplicate last frame
                  |                         |
                  |                         +--> frozen input artifact
                  v
          PyTorch CPU reference
                  |                         +--> reference pulse
                  v
        fixed-shape ONNX export
                  |
          ONNX checker + ORT CPU
                  |
                  +--> parity report + committed manifest
```

The export and validation tools are offline Python utilities. The C++ runtime
and staged Linux package remain Python-free. Model binaries, checkpoint files,
test tensors, and generated ONNX files remain outside Git and outside the CMake
install whitelist.

## Planned project structure

```text
tools/model_export/
  efficientphys_reference.py      model loading and preprocessing contract
  export_efficientphys.py         fixed-shape ONNX export CLI
  validate_efficientphys.py       PyTorch/ORT parity and report CLI
  generate_test_vector.py         deterministic frozen input generation
  requirements.in                 isolated host dependencies
  requirements.lock               exact versions proven by the reference run
  tests/                           unit and integration tests
model_specs/
  efficientphys_pure.json         committed provenance and interface manifest
scripts/
  setup_model_export_macos.sh     creates an isolated, ignored Python 3.12 env
artifacts/model_export/            ignored generated ONNX, arrays and reports
```

The original rPPG repository and the vendored rPPG-Toolbox source are read-only
inputs. No file is created or changed beneath them.

## Dependency policy

- Use an isolated Python 3.12 environment owned by this C++ worktree.
- Use the observed PyTorch 2.12.1 behavior as the reference baseline.
- Add only NumPy, PyTorch, ONNX, ONNX Runtime CPU, PyYAML, and pytest as host
  dependencies; `onnxscript` may be included only if the selected exporter
  requires it.
- Record exact package versions, Python version, platform, and exporter mode in
  every validation report.
- Prefer the legacy fixed-shape exporter with ONNX opset 17 for the first
  reference because QAIRT compatibility is not yet known. If it cannot export
  the official graph, stop with an explicit failure rather than silently
  rewriting the model or changing the interface.

## Commands

The implemented workflow will expose these commands:

```bash
./scripts/setup_model_export_macos.sh

.model-export-venv/bin/python -m pytest tools/model_export/tests -q

.model-export-venv/bin/python tools/model_export/generate_test_vector.py \
  --artifact-dir artifacts/model_export/efficientphys_pure

.model-export-venv/bin/python tools/model_export/export_efficientphys.py \
  --toolbox /Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox \
  --checkpoint /Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth \
  --artifact-dir artifacts/model_export/efficientphys_pure

.model-export-venv/bin/python tools/model_export/validate_efficientphys.py \
  --artifact-dir artifacts/model_export/efficientphys_pure \
  --manifest model_specs/efficientphys_pure.json
```

## Manifest and artifact requirements

The committed JSON manifest uses `schema_version: 1` and records:

- model/checkpoint identity and checkpoint SHA256;
- hashes of the official model, trainer, and inference-config source files;
- preprocessing version and exact tensor contract;
- fixed ONNX opset, input/output names, shapes, layouts, and dtypes;
- generated ONNX SHA256;
- frozen source-window, input-tensor, and PyTorch-output SHA256 values;
- Python and dependency versions;
- parity tolerances and measured results;
- the explicit status `qnn_conversion: not_run`.

The generated artifact directory contains at least:

```text
source_rgb_uint8.npy       [180,72,72,3]
frames_float32.npy         [181,3,72,72]
pytorch_pulse_float32.npy  [180,1]
efficientphys_pure.onnx
onnx_pulse_float32.npy     [180,1]
validation_report.json
```

The deterministic source window is synthetic and contains no personal image
data. It is generated with a versioned algorithm and fixed seed; its purpose is
numerical reproducibility, not accuracy evaluation.

## Error handling

- A checkpoint hash mismatch stops before loading.
- Prefix normalization is accepted only when every checkpoint key begins with
  exactly one `module.`; mixed or unexpected structures fail.
- `load_state_dict` is strict. Missing or unexpected keys fail.
- Wrong input shape, layout, dtype, color order, frame count, or non-finite
  tensor values fail before export.
- ONNX checker or ONNX Runtime failure produces a nonzero exit and a structured
  report; no manifest is updated as successful.
- Parity outside tolerance fails the command and leaves
  `qnn_conversion: not_run`.
- No code falls back to MPS, CUDA, a different checkpoint, a dynamic graph, or
  a rewritten network silently.

## Testing strategy

### Unit tests

- global standardization matches the Toolbox formula, including zero variance;
- RGB channel order and THWC-to-TCHW layout are preserved;
- exactly 180 source frames become 181 tensor frames by duplicating the last;
- 179/181 source windows, incorrect shapes, and non-finite inputs are rejected;
- checkpoint SHA and `module.` prefix handling are strict;
- manifest schema and relative artifact paths are validated.

### Integration tests

- the real public checkpoint loads on CPU with zero missing/unexpected keys;
- PyTorch input `[181,3,72,72]` produces finite `[180,1]` output;
- the fixed ONNX graph passes `onnx.checker`;
- ONNX Runtime CPU accepts the frozen input and produces finite `[180,1]`;
- PyTorch and ONNX output shapes and values satisfy all parity gates;
- the C++ Release/UBSan suite and four-file staging whitelist remain unchanged.

### Parity gates

- maximum absolute output error `<= 1e-4`;
- mean absolute output error `<= 1e-5`;
- Pearson correlation `>= 0.99999` when both outputs have nonzero variance;
- FFT-derived BPM difference `<= 0.1 bpm` for the same 30 FPS output;
- every tensor and metric must be finite.

These are implementation-equivalence gates, not heart-rate accuracy claims.

## Boundaries

### Always

- Run model-export tests before each model tooling commit.
- Run the existing C++ test suite before declaring the phase complete.
- Keep checkpoint, ONNX, `.npy`, `.raw`, and generated reports outside Git and
  outside the deployed package.
- Record SHA256 provenance for every external or generated model artifact.

### Ask first

- Change the 180-source/181-input/180-output contract.
- Rewrite EfficientPhys operators or preprocessing.
- Select a checkpoint other than `PURE_EfficientPhys.pth`.
- Add a training, fine-tuning, calibration, or dataset-download path.

### Never

- Modify the original `/Users/wangjie/Documents/keti/rPPG` repository.
- Commit model weights, personal camera frames, or licensed datasets.
- Present synthetic/fake output as a physiological measurement.
- Mark QNN conversion or Adreno inference successful without target evidence.

## Success criteria

1. A clean Mac setup command creates an isolated export environment without
   modifying the original rPPG environment.
2. The official checkpoint passes hash validation and strict loading.
3. The preprocessing tool deterministically produces a valid 181-frame tensor
   from a 180-frame RGB source window.
4. PyTorch CPU and ONNX Runtime CPU produce finite `[180,1]` outputs that pass
   every parity gate.
5. The committed manifest completely identifies the external checkpoint,
   source implementation, interface, generated ONNX, artifacts, environment,
   and validation result.
6. Generated model artifacts stay ignored and the staged C++ package remains
   Python/model-free.
7. Existing C++ Release and UBSan tests continue to pass.
8. The final report states that QNN conversion, V4L2 Linux behavior, and Adreno
   performance remain unverified until the target SDK and bench are available.
