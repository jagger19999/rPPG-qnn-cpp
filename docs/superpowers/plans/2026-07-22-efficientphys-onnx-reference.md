# EfficientPhys ONNX Reference Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a traceable fixed-shape ONNX reference for the official `PURE_EfficientPhys.pth` checkpoint and prove PyTorch CPU/ONNX Runtime parity without modifying the Python rPPG repository or deploying Python/model artifacts to the Linux bench package.

**Architecture:** Offline Python tools in the independent C++ repository reproduce the Toolbox 180-frame global-standardization path, append the duplicated final frame, load the prefixed checkpoint strictly, export a static `[181,3,72,72] -> [180,1]` graph, and publish hashes plus parity metrics. Generated binaries stay in an ignored artifact directory; only source, tests, a pinned host environment, and a JSON provenance manifest are committed.

**Tech Stack:** Python 3.12, PyTorch 2.12.1, NumPy, ONNX opset 17, ONNX Runtime CPU, pytest, Bash, existing CMake/CTest.

---

## Planned file map

```text
.gitignore                                      ignore export env and artifacts
CMakeLists.txt                                  register hermetic setup smoke test
scripts/setup_model_export_macos.sh             isolated Python 3.12 bootstrap
tests/test_model_export_setup.sh                bootstrap behavior without network
tools/__init__.py                               Python tool namespace
tools/model_export/__init__.py                  model-export package
tools/model_export/requirements.in              direct dependency policy
tools/model_export/requirements.lock            exact tested host versions
tools/model_export/pytest.ini                    test markers
tools/model_export/efficientphys_reference.py   preprocessing/model/provenance core
tools/model_export/generate_test_vector.py      deterministic synthetic source window
tools/model_export/export_efficientphys.py      checkpoint load and ONNX export
tools/model_export/validate_efficientphys.py    ORT parity, BPM and manifest publishing
tools/model_export/tests/test_preprocessing.py  180 -> 181 contract tests
tools/model_export/tests/test_checkpoint.py     checkpoint/prefix/provenance tests
tools/model_export/tests/test_test_vector.py    deterministic artifact tests
tools/model_export/tests/test_export.py         real export integration tests
tools/model_export/tests/test_validation.py     metric/manifest tests
model_specs/efficientphys_pure.json             committed validated manifest
README.md                                       host export and target boundary guide
```

## Fixed external inputs

```text
Toolbox:
/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox

Checkpoint:
/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth

Checkpoint SHA256:
e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49
```

### Task 1: Add the isolated host environment and artifact boundary

**Files:**
- Modify: `.gitignore`
- Modify: `CMakeLists.txt`
- Create: `scripts/setup_model_export_macos.sh`
- Create: `tests/test_model_export_setup.sh`
- Create: `tools/model_export/requirements.in`
- Generate: `tools/model_export/requirements.lock`

- [ ] **Step 1: Write the failing hermetic setup test**

Create `tests/test_model_export_setup.sh`. It supplies a fake Python executable so the test never accesses the network:

```bash
#!/usr/bin/env bash
set -euo pipefail

SOURCE_DIR=${1:?source directory required}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/rppg-model-setup.XXXXXX")
trap 'rm -rf "${tmp}"' EXIT

fake_python="${tmp}/python3.12"
cat >"${fake_python}" <<'PY'
#!/usr/bin/env bash
set -euo pipefail
if [[ ${1:-} == -c ]]; then
  printf '3.12\n'
elif [[ ${1:-} == -m && ${2:-} == venv ]]; then
  mkdir -p "$3/bin"
  cp "$0" "$3/bin/python"
else
  printf '<%s>\n' "$@" >>"${RPPG_SETUP_LOG}"
  if [[ $* == *'pip freeze'* ]]; then
    printf 'numpy==2.4.1\ntorch==2.12.1\n'
  fi
fi
PY
chmod +x "${fake_python}"

log="${tmp}/setup.log"
RPPG_SETUP_LOG="${log}" PYTHON_BIN="${fake_python}" \
  MODEL_EXPORT_VENV="${tmp}/venv" MODEL_EXPORT_REFRESH_LOCK=1 \
  MODEL_EXPORT_LOCK_OUTPUT="${tmp}/requirements.lock" \
  "${SOURCE_DIR}/scripts/setup_model_export_macos.sh"

test -x "${tmp}/venv/bin/python"
grep -Fq '<-m>' "${log}"
grep -Fq '<pip>' "${log}"
grep -Fq 'torch==2.12.1' "${tmp}/requirements.lock"

cat >"${tmp}/python3.14" <<'PY'
#!/usr/bin/env bash
printf '3.14\n'
PY
chmod +x "${tmp}/python3.14"
if PYTHON_BIN="${tmp}/python3.14" MODEL_EXPORT_VENV="${tmp}/bad" \
  "${SOURCE_DIR}/scripts/setup_model_export_macos.sh" >"${tmp}/bad.log" 2>&1; then
  echo 'Python 3.14 unexpectedly accepted' >&2
  exit 1
fi
grep -q 'Python 3.12' "${tmp}/bad.log"
```

- [ ] **Step 2: Register and run the failing test**

Append inside the existing `BUILD_TESTING` block in `CMakeLists.txt`:

```cmake
add_test(NAME model_export_setup
         COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_model_export_setup.sh
                      ${CMAKE_CURRENT_SOURCE_DIR})
```

Run:

```bash
cmake -S . -B build-linux-native -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv@4 -DBUILD_TESTING=ON
ctest --test-dir build-linux-native -R '^model_export_setup$' --output-on-failure
```

Expected: FAIL because `scripts/setup_model_export_macos.sh` does not exist.

- [ ] **Step 3: Add ignored paths and dependency policy**

Append to `.gitignore`:

```gitignore
/.model-export-venv/
/artifacts/model_export/
```

Create `tools/model_export/requirements.in`:

```text
numpy>=2.0,<3
torch==2.12.1
onnx>=1.17,<2
onnxruntime>=1.20,<2
pytest>=8,<10
PyYAML>=6,<7
```

- [ ] **Step 4: Implement the setup script**

Create `scripts/setup_model_export_macos.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail

script_dir=$(CDPATH= cd -P -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_dir=$(CDPATH= cd -P -- "${script_dir}/.." && pwd -P)
python_bin=${PYTHON_BIN:-python3}
venv_dir=${MODEL_EXPORT_VENV:-"${source_dir}/.model-export-venv"}
input_requirements="${source_dir}/tools/model_export/requirements.in"
lock_file=${MODEL_EXPORT_LOCK_OUTPUT:-"${source_dir}/tools/model_export/requirements.lock"}

version=$(${python_bin} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ ${version} != 3.12 ]]; then
  printf 'Python 3.12 is required; got %s\n' "${version}" >&2
  exit 2
fi

"${python_bin}" -m venv "${venv_dir}"
venv_python="${venv_dir}/bin/python"
requirements=${lock_file}
if [[ ${MODEL_EXPORT_REFRESH_LOCK:-0} == 1 || ! -s ${lock_file} ]]; then
  requirements=${input_requirements}
fi
"${venv_python}" -m pip install --upgrade pip
"${venv_python}" -m pip install -r "${requirements}"

if [[ ${MODEL_EXPORT_REFRESH_LOCK:-0} == 1 || ! -s ${lock_file} ]]; then
  lock_tmp="${lock_file}.tmp.$$"
  "${venv_python}" -m pip freeze | LC_ALL=C sort >"${lock_tmp}"
  mv -- "${lock_tmp}" "${lock_file}"
fi
printf 'Model export environment ready: %s\n' "${venv_dir}"
```

- [ ] **Step 5: Run the hermetic test, then resolve the real lock**

```bash
chmod +x scripts/setup_model_export_macos.sh tests/test_model_export_setup.sh
ctest --test-dir build-linux-native -R '^model_export_setup$' --output-on-failure

PYTHON_BIN=/Users/wangjie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
MODEL_EXPORT_REFRESH_LOCK=1 ./scripts/setup_model_export_macos.sh
.model-export-venv/bin/python -c \
  'import torch, onnx, onnxruntime; print(torch.__version__, onnx.__version__, onnxruntime.__version__)'
```

Expected: hermetic test PASS; real command prints Python dependency versions and creates a nonempty pinned `requirements.lock`.

- [ ] **Step 6: Commit**

```bash
git add .gitignore CMakeLists.txt scripts/setup_model_export_macos.sh \
  tests/test_model_export_setup.sh tools/model_export/requirements.in \
  tools/model_export/requirements.lock
git commit -m "chore: isolate EfficientPhys export environment"
```

### Task 2: Implement the exact 180-to-181 preprocessing contract

**Files:**
- Create: `tools/__init__.py`
- Create: `tools/model_export/__init__.py`
- Create: `tools/model_export/pytest.ini`
- Create: `tools/model_export/efficientphys_reference.py`
- Create: `tools/model_export/tests/test_preprocessing.py`

- [ ] **Step 1: Write failing preprocessing tests**

Create `tools/model_export/tests/test_preprocessing.py`:

```python
import numpy as np
import pytest

from tools.model_export.efficientphys_reference import prepare_model_input


def test_global_standardization_and_last_frame_duplication():
    source = np.arange(180 * 72 * 72 * 3, dtype=np.uint32)
    source = (source % 256).astype(np.uint8).reshape(180, 72, 72, 3)

    result = prepare_model_input(source)

    expected = source.astype(np.float64)
    expected = (expected - expected.mean()) / expected.std()
    expected = expected.astype(np.float32).transpose(0, 3, 1, 2)
    assert result.shape == (181, 3, 72, 72)
    assert result.dtype == np.float32
    np.testing.assert_allclose(result[:180], expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(result[180], result[179])


def test_zero_variance_becomes_finite_zeros():
    result = prepare_model_input(np.full((180, 72, 72, 3), 17, dtype=np.uint8))
    assert np.isfinite(result).all()
    assert np.count_nonzero(result) == 0


@pytest.mark.parametrize("shape", [(179, 72, 72, 3), (181, 72, 72, 3), (180, 72, 72, 1)])
def test_wrong_source_shape_is_rejected(shape):
    with pytest.raises(ValueError, match="180, 72, 72, 3"):
        prepare_model_input(np.zeros(shape, dtype=np.uint8))


def test_non_uint8_source_is_rejected():
    with pytest.raises(TypeError, match="uint8"):
        prepare_model_input(np.zeros((180, 72, 72, 3), dtype=np.float32))
```

- [ ] **Step 2: Run the test to verify RED**

```bash
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_preprocessing.py -q
```

Expected: collection FAIL because `efficientphys_reference` does not exist.

- [ ] **Step 3: Implement the preprocessing core**

Create empty `tools/__init__.py` and `tools/model_export/__init__.py`. Create
`tools/model_export/pytest.ini`:

```ini
[pytest]
markers =
    integration: requires the public checkpoint and ONNX Runtime
```

Create the initial `tools/model_export/efficientphys_reference.py`:

```python
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

SOURCE_SHAPE = (180, 72, 72, 3)
MODEL_INPUT_SHAPE = (181, 3, 72, 72)
MODEL_OUTPUT_SHAPE = (180, 1)
FRAME_RATE = 30.0
FRAME_DEPTH = 10
EXPECTED_CHECKPOINT_SHA256 = (
    "e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_model_input(source_rgb: np.ndarray) -> np.ndarray:
    if source_rgb.dtype != np.uint8:
        raise TypeError("source RGB frames must use uint8")
    if source_rgb.shape != SOURCE_SHAPE:
        raise ValueError(f"source RGB shape must be {SOURCE_SHAPE}")
    values = source_rgb.astype(np.float64)
    deviation = float(values.std())
    if deviation == 0.0 or not np.isfinite(deviation):
        values = np.zeros_like(values)
    else:
        values = (values - float(values.mean())) / deviation
    values[~np.isfinite(values)] = 0.0
    tchw = values.astype(np.float32).transpose(0, 3, 1, 2)
    result = np.concatenate((tchw, tchw[-1:]), axis=0)
    if result.shape != MODEL_INPUT_SHAPE or not np.isfinite(result).all():
        raise ValueError("preprocessed EfficientPhys tensor is invalid")
    return np.ascontiguousarray(result)
```

- [ ] **Step 4: Run GREEN and commit**

```bash
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_preprocessing.py -q
git add tools/__init__.py tools/model_export
git commit -m "feat: reproduce EfficientPhys preprocessing contract"
```

Expected: all preprocessing tests PASS.

### Task 3: Add strict official checkpoint loading and provenance

**Files:**
- Modify: `tools/model_export/efficientphys_reference.py`
- Create: `tools/model_export/tests/test_checkpoint.py`

- [ ] **Step 1: Write failing prefix and real-checkpoint tests**

Create `tools/model_export/tests/test_checkpoint.py`:

```python
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from tools.model_export.efficientphys_reference import (
    EXPECTED_CHECKPOINT_SHA256,
    load_official_model,
    normalize_module_prefix,
    sha256_file,
)


def test_normalize_module_prefix_requires_one_prefix_on_every_key():
    state = OrderedDict((f"module.layer{n}", torch.tensor([n])) for n in range(2))
    assert list(normalize_module_prefix(state)) == ["layer0", "layer1"]

    with pytest.raises(ValueError, match="every key"):
        normalize_module_prefix(OrderedDict([("module.a", torch.tensor(1)), ("b", torch.tensor(2))]))
    with pytest.raises(ValueError, match="exactly one"):
        normalize_module_prefix(OrderedDict([("module.module.a", torch.tensor(1))]))


@pytest.mark.integration
def test_public_checkpoint_loads_strictly_and_outputs_180_samples():
    toolbox = Path(__import__("os").environ["RPPG_TOOLBOX_PATH"])
    checkpoint = Path(__import__("os").environ["RPPG_EFFICIENTPHYS_CHECKPOINT"])
    assert sha256_file(checkpoint) == EXPECTED_CHECKPOINT_SHA256
    model = load_official_model(toolbox, checkpoint)
    with torch.inference_mode():
        output = model(torch.zeros((181, 3, 72, 72), dtype=torch.float32))
    assert tuple(output.shape) == (180, 1)
    assert torch.isfinite(output).all()
```

- [ ] **Step 2: Verify RED**

```bash
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_checkpoint.py -q -m 'not integration'
```

Expected: FAIL because prefix/model functions are undefined.

- [ ] **Step 3: Implement strict loading**

Add to `efficientphys_reference.py`:

```python
import importlib.util
from collections import OrderedDict
from collections.abc import Mapping
from types import ModuleType

import torch


def normalize_module_prefix(state: Mapping[str, torch.Tensor]) -> OrderedDict:
    if not state or not all(isinstance(key, str) for key in state):
        raise ValueError("checkpoint must contain a nonempty string-keyed state dict")
    if not all(key.startswith("module.") for key in state):
        raise ValueError("every key must begin with module.")
    if any(key.startswith("module.module.") for key in state):
        raise ValueError("checkpoint keys must contain exactly one module. prefix")
    return OrderedDict((key[len("module."):], value) for key, value in state.items())


def _load_model_module(toolbox: Path) -> ModuleType:
    source = toolbox / "neural_methods/model/EfficientPhys.py"
    if not source.is_file():
        raise FileNotFoundError(f"EfficientPhys source not found: {source}")
    spec = importlib.util.spec_from_file_location("rppg_toolbox_efficientphys", source)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import EfficientPhys source: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_model(toolbox: Path, checkpoint: Path) -> torch.nn.Module:
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must be a state dict")
    clean_state = normalize_module_prefix(state)
    model_class = _load_model_module(toolbox).EfficientPhys
    model = model_class(frame_depth=FRAME_DEPTH, img_size=72)
    model.load_state_dict(clean_state, strict=True)
    model.eval()
    return model.cpu()
```

- [ ] **Step 4: Run unit and real integration tests**

```bash
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_checkpoint.py -q -m 'not integration'

RPPG_TOOLBOX_PATH=/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox \
RPPG_EFFICIENTPHYS_CHECKPOINT=/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth \
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_checkpoint.py -q -m integration
```

Expected: both commands PASS; integration produces finite `[180,1]`.

- [ ] **Step 5: Commit**

```bash
git add tools/model_export/efficientphys_reference.py \
  tools/model_export/tests/test_checkpoint.py
git commit -m "feat: load official EfficientPhys checkpoint strictly"
```

### Task 4: Generate deterministic frozen input artifacts

**Files:**
- Create: `tools/model_export/generate_test_vector.py`
- Create: `tools/model_export/tests/test_test_vector.py`
- Modify: `.gitignore`

- [ ] **Step 1: Write the failing deterministic-vector test**

Create `tools/model_export/tests/test_test_vector.py`:

```python
import hashlib
import numpy as np

from tools.model_export.generate_test_vector import build_source_window
from tools.model_export.efficientphys_reference import prepare_model_input


def raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_synthetic_source_and_tensor_are_bit_reproducible():
    source = build_source_window()
    frames = prepare_model_input(source)
    assert raw_sha256(source) == "dca274b1e509762f17a2dff152fa0fc3fa87c0c6f2726db284792d3494350b46"
    assert raw_sha256(frames) == "b685ed04d5b6a96bbdd2f97fed6cc04a8a2eb35c38b7ebc820eb1a37af2d2f40"
```

- [ ] **Step 2: Verify RED**

```bash
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_test_vector.py -q
```

Expected: FAIL because `generate_test_vector` does not exist.

- [ ] **Step 3: Implement the generator CLI**

Create `tools/model_export/generate_test_vector.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from tools.model_export.efficientphys_reference import prepare_model_input

GENERATOR_SEED = 20260722


def build_source_window() -> np.ndarray:
    t = np.arange(180, dtype=np.uint32)[:, None, None, None]
    y = np.arange(72, dtype=np.uint32)[None, :, None, None]
    x = np.arange(72, dtype=np.uint32)[None, None, :, None]
    c = np.arange(3, dtype=np.uint32)[None, None, None, :]
    return ((GENERATOR_SEED + t * 17 + y * 13 + x * 7 + c * 53 + (t * y) % 251) % 256).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    source = build_source_window()
    frames = prepare_model_input(source)
    np.save(args.artifact_dir / "source_rgb_uint8.npy", source, allow_pickle=False)
    np.save(args.artifact_dir / "frames_float32.npy", frames, allow_pickle=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests and generate ignored artifacts**

```bash
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_test_vector.py -q
.model-export-venv/bin/python tools/model_export/generate_test_vector.py \
  --artifact-dir artifacts/model_export/efficientphys_pure
git status --short
```

Expected: test PASS; two `.npy` files exist but do not appear in Git status.

- [ ] **Step 5: Commit**

```bash
git add tools/model_export/generate_test_vector.py \
  tools/model_export/tests/test_test_vector.py .gitignore
git commit -m "feat: freeze deterministic EfficientPhys input vector"
```

### Task 5: Export and check the fixed-shape ONNX graph

**Files:**
- Create: `tools/model_export/export_efficientphys.py`
- Create: `tools/model_export/tests/test_export.py`

- [ ] **Step 1: Write the failing real-export integration test**

Create `tools/model_export/tests/test_export.py`:

```python
from pathlib import Path
import os

import numpy as np
import onnx
import pytest

from tools.model_export.export_efficientphys import export_reference


@pytest.mark.integration
def test_fixed_graph_exports_with_stable_interface(tmp_path):
    toolbox = Path(os.environ["RPPG_TOOLBOX_PATH"])
    checkpoint = Path(os.environ["RPPG_EFFICIENTPHYS_CHECKPOINT"])
    frames = np.load(Path(os.environ["RPPG_MODEL_ARTIFACT_DIR"]) / "frames_float32.npy", allow_pickle=False)
    paths = export_reference(toolbox, checkpoint, frames, tmp_path)
    model = onnx.load(paths.onnx)
    onnx.checker.check_model(model)
    assert model.graph.input[0].name == "frames"
    assert model.graph.output[0].name == "pulse"
    assert paths.pytorch_output.is_file()
    assert np.load(paths.pytorch_output, allow_pickle=False).shape == (180, 1)
```

- [ ] **Step 2: Verify RED**

Run the integration command from Task 3 with the additional environment value:

```bash
RPPG_MODEL_ARTIFACT_DIR=$PWD/artifacts/model_export/efficientphys_pure \
.model-export-venv/bin/python -m pytest tools/model_export/tests/test_export.py -q
```

Expected: collection FAIL because `export_efficientphys` does not exist.

- [ ] **Step 3: Implement fixed export with atomic publication**

Create `tools/model_export/export_efficientphys.py` with:

```python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import torch

from tools.model_export.efficientphys_reference import MODEL_INPUT_SHAPE, MODEL_OUTPUT_SHAPE, load_official_model


@dataclass(frozen=True)
class ExportPaths:
    onnx: Path
    pytorch_output: Path


def export_reference(toolbox: Path, checkpoint: Path, frames: np.ndarray, artifact_dir: Path) -> ExportPaths:
    if frames.shape != MODEL_INPUT_SHAPE or frames.dtype != np.float32 or not np.isfinite(frames).all():
        raise ValueError("frames must be finite float32 [181,3,72,72]")
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model = load_official_model(toolbox, checkpoint)
    tensor = torch.from_numpy(np.ascontiguousarray(frames))
    with torch.inference_mode():
        output = model(tensor).detach().cpu().numpy().astype(np.float32)
    if output.shape != MODEL_OUTPUT_SHAPE or not np.isfinite(output).all():
        raise ValueError("PyTorch output is invalid")
    output_path = artifact_dir / "pytorch_pulse_float32.npy"
    np.save(output_path, output, allow_pickle=False)

    onnx_path = artifact_dir / "efficientphys_pure.onnx"
    temporary = artifact_dir / "efficientphys_pure.onnx.tmp"
    try:
        torch.onnx.export(
            model,
            tensor,
            temporary,
            input_names=["frames"],
            output_names=["pulse"],
            opset_version=17,
            do_constant_folding=True,
            dynamo=False,
        )
        checked = onnx.load(temporary)
        onnx.checker.check_model(checked)
        temporary.replace(onnx_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return ExportPaths(onnx=onnx_path, pytorch_output=output_path)
```

Complete the file with this CLI. It returns nonzero naturally on exceptions and
does not add a fallback exporter:

```python
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolbox", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    frames_path = args.artifact_dir / "frames_float32.npy"
    report_path = args.artifact_dir / "export_report.json"
    try:
        frames = np.load(frames_path, allow_pickle=False)
        paths = export_reference(args.toolbox, args.checkpoint, frames, args.artifact_dir)
    except Exception as error:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        temporary = report_path.with_name(report_path.name + ".tmp")
        temporary.write_text(
            json.dumps({"schema_version": 1, "passed": False, "error": str(error)}, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(report_path)
        raise
    temporary = report_path.with_name(report_path.name + ".tmp")
    temporary.write_text(
        json.dumps({"schema_version": 1, "passed": True}, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(paths.onnx)
    print(paths.pytorch_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run real export and integration test**

```bash
.model-export-venv/bin/python tools/model_export/export_efficientphys.py \
  --toolbox /Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox \
  --checkpoint /Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth \
  --artifact-dir artifacts/model_export/efficientphys_pure

RPPG_TOOLBOX_PATH=/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox \
RPPG_EFFICIENTPHYS_CHECKPOINT=/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth \
RPPG_MODEL_ARTIFACT_DIR=$PWD/artifacts/model_export/efficientphys_pure \
.model-export-venv/bin/python -m pytest tools/model_export/tests/test_export.py -q
```

Expected: export succeeds; integration PASS; ONNX input/output names and shapes are fixed.

- [ ] **Step 5: Commit**

```bash
git add tools/model_export/export_efficientphys.py \
  tools/model_export/tests/test_export.py
git commit -m "feat: export fixed EfficientPhys ONNX reference"
```

### Task 6: Validate ONNX Runtime parity and publish the manifest

**Files:**
- Create: `tools/model_export/validate_efficientphys.py`
- Create: `tools/model_export/tests/test_validation.py`
- Create: `model_specs/efficientphys_pure.json`

- [ ] **Step 1: Write failing metric and publication tests**

Create `tools/model_export/tests/test_validation.py`:

```python
import json
import numpy as np
import pytest

from tools.model_export.validate_efficientphys import compare_outputs, publish_json_atomic


def test_parity_metrics_accept_close_outputs_and_reject_drift():
    reference = np.sin(np.linspace(0, 12, 180, dtype=np.float32)).reshape(180, 1)
    close = reference + np.float32(1e-6)
    metrics = compare_outputs(reference, close, fps=30.0)
    assert metrics["passed"] is True
    assert metrics["max_abs_error"] <= 1e-4

    drift = reference + np.linspace(0, 0.1, 180, dtype=np.float32).reshape(180, 1)
    assert compare_outputs(reference, drift, fps=30.0)["passed"] is False


def test_atomic_json_publication_never_leaves_temporary_file(tmp_path):
    destination = tmp_path / "manifest.json"
    publish_json_atomic(destination, {"schema_version": 1})
    assert json.loads(destination.read_text()) == {"schema_version": 1}
    assert not (tmp_path / "manifest.json.tmp").exists()


def test_nonfinite_output_is_rejected():
    values = np.zeros((180, 1), dtype=np.float32)
    values[0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        compare_outputs(values, np.zeros_like(values), fps=30.0)
```

- [ ] **Step 2: Verify RED**

```bash
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_validation.py -q
```

Expected: collection FAIL because the validation module does not exist.

- [ ] **Step 3: Implement metrics, ORT execution and atomic JSON**

Create `tools/model_export/validate_efficientphys.py` with the complete
validation and manifest publisher:

```python
from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

from tools.model_export.efficientphys_reference import (
    EXPECTED_CHECKPOINT_SHA256,
    FRAME_RATE,
    sha256_file,
)


def fft_bpm(waveform: np.ndarray, fps: float) -> float:
    signal = waveform.reshape(-1).astype(np.float64)
    signal -= signal.mean()
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / fps)
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (frequencies >= 0.7) & (frequencies <= 3.0)
    if not mask.any() or not np.isfinite(power[mask]).all():
        raise ValueError("cannot derive finite band-limited BPM")
    return float(frequencies[mask][np.argmax(power[mask])] * 60.0)


def compare_outputs(reference: np.ndarray, candidate: np.ndarray, fps: float) -> dict:
    if reference.shape != (180, 1) or candidate.shape != (180, 1):
        raise ValueError("outputs must have shape [180,1]")
    if not np.isfinite(reference).all() or not np.isfinite(candidate).all():
        raise ValueError("outputs must be finite")
    difference = np.abs(reference.astype(np.float64) - candidate.astype(np.float64))
    ref_std = float(reference.std())
    candidate_std = float(candidate.std())
    pearson = None
    pearson_pass = True
    if ref_std > 0.0 and candidate_std > 0.0:
        pearson = float(np.corrcoef(reference.reshape(-1), candidate.reshape(-1))[0, 1])
        pearson_pass = pearson >= 0.99999
    bpm_error = abs(fft_bpm(reference, fps) - fft_bpm(candidate, fps))
    maximum = float(difference.max())
    mean = float(difference.mean())
    return {
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "pearson": pearson,
        "bpm_error": bpm_error,
        "passed": maximum <= 1e-4 and mean <= 1e-5 and pearson_pass and bpm_error <= 0.1,
    }


def publish_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_onnx(onnx_path: Path, frames: np.ndarray) -> np.ndarray:
    if frames.shape != (181, 3, 72, 72) or frames.dtype != np.float32:
        raise ValueError("ONNX input must be float32 [181,3,72,72]")
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    if len(session.get_inputs()) != 1 or session.get_inputs()[0].name != "frames":
        raise ValueError("ONNX graph must expose exactly one frames input")
    if len(session.get_outputs()) != 1 or session.get_outputs()[0].name != "pulse":
        raise ValueError("ONNX graph must expose exactly one pulse output")
    output = np.asarray(session.run(["pulse"], {"frames": frames})[0])
    if output.dtype != np.float32 or output.shape != (180, 1) or not np.isfinite(output).all():
        raise ValueError("ONNX output must be finite float32 [180,1]")
    return output


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "torch", "onnx", "onnxruntime")
    }


def artifact_entry(artifact_dir: Path, name: str, array: np.ndarray) -> dict:
    path = artifact_dir / name
    return {
        "file": name,
        "sha256": sha256_file(path),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
    }


def validate_and_publish(toolbox: Path, checkpoint: Path, artifact_dir: Path, manifest_path: Path) -> dict:
    source = np.load(artifact_dir / "source_rgb_uint8.npy", allow_pickle=False)
    frames = np.load(artifact_dir / "frames_float32.npy", allow_pickle=False)
    pytorch_output = np.load(artifact_dir / "pytorch_pulse_float32.npy", allow_pickle=False)
    if source.shape != (180, 72, 72, 3) or source.dtype != np.uint8:
        raise ValueError("source artifact must be uint8 [180,72,72,3]")
    if sha256_file(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA256 mismatch")
    onnx_path = artifact_dir / "efficientphys_pure.onnx"
    onnx_output = run_onnx(onnx_path, frames)
    np.save(artifact_dir / "onnx_pulse_float32.npy", onnx_output, allow_pickle=False)
    validation = compare_outputs(pytorch_output, onnx_output, FRAME_RATE)

    report = {
        "schema_version": 1,
        "passed": validation["passed"],
        "validation": validation,
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "packages": package_versions(),
        },
    }
    publish_json_atomic(artifact_dir / "validation_report.json", report)
    if not validation["passed"]:
        raise RuntimeError("PyTorch and ONNX outputs failed parity gates")

    official_files = {
        "model": "neural_methods/model/EfficientPhys.py",
        "trainer": "neural_methods/trainer/EfficientPhysTrainer.py",
        "inference_config": "configs/infer_configs/PURE_UBFC-rPPG_EFFICIENTPHYS.yaml",
    }
    manifest = {
        "schema_version": 1,
        "model_name": "EfficientPhys",
        "checkpoint": {"filename": checkpoint.name, "sha256": sha256_file(checkpoint)},
        "official_sources": {
            name: {"file": relative, "sha256": sha256_file(toolbox / relative)}
            for name, relative in official_files.items()
        },
        "preprocessing": {
            "source_shape": [180, 72, 72, 3],
            "source_dtype": "uint8",
            "source_layout": "THWC",
            "color_order": "RGB",
            "standardization": "one population mean/std over complete source window",
            "append_last_frame": True,
            "frame_rate": 30.0,
        },
        "test_vector": {"algorithm_version": 1, "seed": 20260722},
        "interface": {
            "input": {"name": "frames", "shape": [181, 3, 72, 72], "dtype": "float32", "layout": "TCHW"},
            "output": {"name": "pulse", "shape": [180, 1], "dtype": "float32"},
            "frame_depth": 10,
        },
        "onnx": {"file": onnx_path.name, "sha256": sha256_file(onnx_path), "opset": 17},
        "artifacts": {
            "source": artifact_entry(artifact_dir, "source_rgb_uint8.npy", source),
            "frames": artifact_entry(artifact_dir, "frames_float32.npy", frames),
            "pytorch_output": artifact_entry(artifact_dir, "pytorch_pulse_float32.npy", pytorch_output),
            "onnx_output": artifact_entry(artifact_dir, "onnx_pulse_float32.npy", onnx_output),
        },
        "environment": report["environment"],
        "validation": {"passed": True, **validation},
        "qnn_conversion": "not_run",
    }
    publish_json_atomic(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolbox", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        validate_and_publish(args.toolbox, args.checkpoint, args.artifact_dir, args.manifest)
    except Exception as error:
        publish_json_atomic(
            args.artifact_dir / "validation_report.json",
            {"schema_version": 1, "passed": False, "error": str(error)},
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

The structured failure report is written before the exception propagates, while
the committed manifest is published only after every parity gate passes.

- [ ] **Step 4: Run unit tests and real validation**

```bash
.model-export-venv/bin/python -m pytest \
  tools/model_export/tests/test_validation.py -q

.model-export-venv/bin/python tools/model_export/validate_efficientphys.py \
  --toolbox /Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox \
  --checkpoint /Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth \
  --artifact-dir artifacts/model_export/efficientphys_pure \
  --manifest model_specs/efficientphys_pure.json
```

Expected: exit 0; report and manifest say `passed: true`; output shapes are `[180,1]`; QNN status remains `not_run`.

- [ ] **Step 5: Validate the committed manifest has no absolute private artifact paths**

```bash
.model-export-venv/bin/python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path('model_specs/efficientphys_pure.json').read_text())
assert p['schema_version'] == 1
assert p['qnn_conversion'] == 'not_run'
assert p['validation']['passed'] is True
assert '/Users/' not in json.dumps(p)
PY
```

- [ ] **Step 6: Commit**

```bash
git add tools/model_export/validate_efficientphys.py \
  tools/model_export/tests/test_validation.py model_specs/efficientphys_pure.json
git commit -m "feat: validate EfficientPhys ONNX parity"
```

### Task 7: Document and run the complete phase gate

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add the offline model-export section to README**

Append this section, keeping the existing Linux operator guide intact:

````markdown
## Offline EfficientPhys ONNX reference (host only)

The model-export tools run only on the development host and are not installed
by CMake. They use the public `PURE_EfficientPhys.pth` checkpoint with SHA256
`e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49`.
The official contract is 180 RGB frames at 30 FPS and 72 x 72, global window
standardization, TCHW conversion, and one duplicate final frame. The ONNX
interface is therefore `[181,3,72,72] -> [180,1]`.

```bash
PYTHON_BIN=/Users/wangjie/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  ./scripts/setup_model_export_macos.sh
.model-export-venv/bin/python -m pytest tools/model_export/tests -q -m 'not integration'
.model-export-venv/bin/python tools/model_export/generate_test_vector.py \
  --artifact-dir artifacts/model_export/efficientphys_pure
.model-export-venv/bin/python tools/model_export/export_efficientphys.py \
  --toolbox /path/to/rPPG-Toolbox --checkpoint /path/to/PURE_EfficientPhys.pth \
  --artifact-dir artifacts/model_export/efficientphys_pure
.model-export-venv/bin/python tools/model_export/validate_efficientphys.py \
  --toolbox /path/to/rPPG-Toolbox --checkpoint /path/to/PURE_EfficientPhys.pth \
  --artifact-dir artifacts/model_export/efficientphys_pure \
  --manifest model_specs/efficientphys_pure.json
```

Generated ONNX, NumPy arrays and reports remain under the ignored
`artifacts/model_export/` directory. The committed
`model_specs/efficientphys_pure.json` is the source of truth for hashes,
versions and measured parity. This reference does not mean QAIRT conversion,
Adreno execution, Linux V4L2 behavior or physiological accuracy has been
validated.
````

- [ ] **Step 2: Run all Python unit and integration tests**

```bash
.model-export-venv/bin/python -m pytest tools/model_export/tests -q -m 'not integration'

RPPG_TOOLBOX_PATH=/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox \
RPPG_EFFICIENTPHYS_CHECKPOINT=/Users/wangjie/Documents/keti/rPPG/docs/code_repos/official/rPPG-Toolbox/final_model_release/PURE_EfficientPhys.pth \
RPPG_MODEL_ARTIFACT_DIR=$PWD/artifacts/model_export/efficientphys_pure \
.model-export-venv/bin/python -m pytest tools/model_export/tests -q -m integration
```

Expected: both suites PASS with no skipped selected tests.

- [ ] **Step 3: Run full C++ Release regression and package whitelist**

```bash
CMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv@4 ./scripts/build_linux.sh native
find stage/rppg-qnn -type f | LC_ALL=C sort
```

Expected: all CTests PASS; staged package contains only:

```text
stage/rppg-qnn/bin/rppg_qnn_live
stage/rppg-qnn/bin/run_rppg_qnn.sh
stage/rppg-qnn/share/rppg-qnn/README.md
stage/rppg-qnn/share/rppg-qnn/config/runtime-defaults.env
```

- [ ] **Step 4: Run UBSan-only regression**

```bash
cmake -S . -B build-model-export-ubsan \
  -DCMAKE_PREFIX_PATH=/opt/homebrew/opt/opencv@4 \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=ON \
  '-DCMAKE_CXX_FLAGS=-fsanitize=undefined -fno-omit-frame-pointer' \
  '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=undefined' \
  '-DCMAKE_SHARED_LINKER_FLAGS=-fsanitize=undefined'
cmake --build build-model-export-ubsan --parallel
ctest --test-dir build-model-export-ubsan --output-on-failure
```

Expected: all tests PASS without an undefined-behavior report.

- [ ] **Step 5: Verify isolation and ignored artifacts**

```bash
git status --short --branch
git -C /Users/wangjie/Documents/keti/rPPG status --short --branch
git ls-files | grep -E '\.(pth|onnx|npy|raw|dlc)$' && exit 1 || true
```

Expected: only planned README/spec changes remain before the final commit; the
original repository still shows only its pre-existing untracked TSCAN
checkpoint; no generated model binary is tracked.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md docs/superpowers/specs/2026-07-22-efficientphys-onnx-reference-design.md
git commit -m "docs: explain EfficientPhys reference export"
```

- [ ] **Step 7: Request independent code and numerical review**

The reviewer must inspect checkpoint provenance, preprocessing equivalence,
static ONNX shapes, parity calculations, manifest paths/hashes, failure
behavior, package exclusion, and original-repository isolation. Any Critical or
Important finding is fixed with a failing regression test before completion.

## Completion boundary

This plan is complete only when the Mac reference package passes all Python,
ONNX, C++ Release, and UBSan gates and the committed manifest is traceable.
Completion does **not** mean EfficientPhys runs through QAIRT/QNN, V4L2 works on
the target Linux image, or Adreno latency/accuracy has been measured. Those are
the next hardware-backed plan.
