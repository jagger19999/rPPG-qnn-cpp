from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from tools.model_export import validate_efficientphys as validator


INPUT_SHAPE = (181, 3, 72, 72)
OUTPUT_SHAPE = (180, 1)


def _pulse() -> np.ndarray:
    time = np.arange(180, dtype=np.float32) / np.float32(30.0)
    return np.sin(np.float32(2.0 * np.pi * 1.5) * time).reshape(OUTPUT_SHAPE)


def _raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_fft_bpm_uses_float64_mean_detrended_band_limited_spectrum():
    time = np.arange(180, dtype=np.float64) / 30.0
    signal = (22.0 + np.sin(2.0 * np.pi * 1.5 * time)).reshape(OUTPUT_SHAPE)

    assert validator.fft_bpm(signal, 30.0) == pytest.approx(90.0)


@pytest.mark.parametrize("fps", [0.0, -1.0, np.nan, np.inf])
def test_fft_bpm_rejects_nonpositive_or_nonfinite_fps(fps):
    with pytest.raises(ValueError, match="fps.*finite and positive"):
        validator.fft_bpm(_pulse(), fps)


@pytest.mark.parametrize(
    "waveform",
    [
        np.zeros(OUTPUT_SHAPE, dtype=np.float32),
        np.full(OUTPUT_SHAPE, np.nan, dtype=np.float32),
    ],
)
def test_fft_bpm_rejects_nonfinite_or_zero_band_energy(waveform):
    with pytest.raises(ValueError, match="finite.*positive.*band"):
        validator.fft_bpm(waveform, 30.0)


def test_compare_outputs_accepts_close_outputs_and_returns_explicit_gates():
    reference = _pulse()
    candidate = reference + np.float32(1e-6)

    metrics = validator.compare_outputs(reference, candidate, fps=30.0)

    assert metrics["passed"] is True
    assert metrics["max_abs_error"] <= 1e-4
    assert metrics["mean_abs_error"] <= 1e-5
    assert metrics["pearson"] >= 0.99999
    assert metrics["bpm_error"] <= 0.1
    assert metrics["thresholds"] == {
        "max_abs_error": 1e-4,
        "mean_abs_error": 1e-5,
        "pearson": 0.99999,
        "bpm_error": 0.1,
    }


def test_compare_outputs_rejects_drift_by_returning_failed_metrics():
    reference = _pulse()
    drift = reference + np.linspace(0, 0.1, 180, dtype=np.float32).reshape(OUTPUT_SHAPE)

    metrics = validator.compare_outputs(reference, drift, fps=30.0)

    assert metrics["passed"] is False
    assert metrics["max_abs_error"] > metrics["thresholds"]["max_abs_error"]


@pytest.mark.parametrize(
    ("reference", "candidate", "message"),
    [
        (np.zeros((180,), dtype=np.float32), _pulse(), "shape"),
        (_pulse(), np.zeros((1, 180), dtype=np.float32), "shape"),
        (_pulse().astype(np.float64), _pulse(), "dtype.*float32"),
        (_pulse(), _pulse().astype(np.float64), "dtype.*float32"),
        (np.full(OUTPUT_SHAPE, np.nan, dtype=np.float32), _pulse(), "finite"),
        (_pulse(), np.full(OUTPUT_SHAPE, np.inf, dtype=np.float32), "finite"),
        (np.ones(OUTPUT_SHAPE, dtype=np.float32), _pulse(), "standard deviation"),
        (_pulse(), np.ones(OUTPUT_SHAPE, dtype=np.float32), "standard deviation"),
    ],
)
def test_compare_outputs_rejects_invalid_or_degenerate_outputs(
    reference, candidate, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        validator.compare_outputs(reference, candidate, fps=30.0)


def test_publish_json_atomic_uses_unique_same_directory_temps(tmp_path, monkeypatch):
    destinations = []
    real_replace = os.replace

    def recording_replace(source, destination):
        destinations.append(Path(source))
        return real_replace(source, destination)

    monkeypatch.setattr(validator.os, "replace", recording_replace)
    path = tmp_path / "nested" / "manifest.json"
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(
            executor.map(
                lambda value: validator.publish_json_atomic(path, {"v": value}),
                range(2),
            )
        )

    assert json.loads(path.read_text())["v"] in (0, 1)
    assert len({item.name for item in destinations}) == 2
    assert all(item.parent == path.parent for item in destinations)
    assert not list(path.parent.glob(f".{path.name}.*.tmp"))


def test_publish_json_atomic_publishes_world_readable_file(tmp_path):
    destination = tmp_path / "manifest.json"

    validator.publish_json_atomic(destination, {"schema_version": 1})

    assert destination.stat().st_mode & 0o777 == 0o644


def test_publish_json_atomic_write_failure_preserves_final_and_cleans_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "manifest.json"
    destination.write_text("old")

    def failing_dump(_payload, output, **_kwargs):
        output.write("partial")
        raise OSError("disk full")

    monkeypatch.setattr(validator.json, "dump", failing_dump)
    with pytest.raises(OSError, match="disk full"):
        validator.publish_json_atomic(destination, {"new": True})

    assert destination.read_text() == "old"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


def test_publish_json_atomic_replace_failure_preserves_final_and_cleans_temp(
    tmp_path, monkeypatch
):
    destination = tmp_path / "manifest.json"
    destination.write_text("old")

    def failing_replace(_source, _destination):
        raise OSError("replace failed")

    monkeypatch.setattr(validator.os, "replace", failing_replace)
    with pytest.raises(OSError, match="replace failed"):
        validator.publish_json_atomic(destination, {"new": True})

    assert destination.read_text() == "old"
    assert not list(tmp_path.glob(f".{destination.name}.*.tmp"))


class _Node:
    def __init__(self, name, shape, tensor_type="tensor(float)"):
        self.name = name
        self.shape = shape
        self.type = tensor_type


class _Session:
    providers = ["CPUExecutionProvider"]
    inputs = [_Node("frames", list(INPUT_SHAPE))]
    outputs = [_Node("pulse", list(OUTPUT_SHAPE))]
    result = _pulse()
    requested_providers = None

    def __init__(self, _path, providers):
        type(self).requested_providers = providers

    def get_inputs(self):
        return self.inputs

    def get_outputs(self):
        return self.outputs

    def get_providers(self):
        return self.providers

    def run(self, output_names, feeds):
        assert output_names == ["pulse"]
        assert list(feeds) == ["frames"]
        return [self.result]


def test_run_onnx_requests_and_records_cpu_provider(monkeypatch, tmp_path):
    monkeypatch.setattr(validator.ort, "InferenceSession", _Session)

    result, runtime = validator._run_onnx_with_metadata(
        tmp_path / "model.onnx", np.zeros(INPUT_SHAPE, dtype=np.float32)
    )

    assert np.array_equal(result, _pulse())
    assert _Session.requested_providers == ["CPUExecutionProvider"]
    assert runtime["requested_provider"] == "CPUExecutionProvider"
    assert runtime["providers"] == ["CPUExecutionProvider"]
    assert runtime["inference_seconds"] >= 0.0


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        (np.zeros((180, 3, 72, 72), dtype=np.float32), "shape"),
        (np.zeros(INPUT_SHAPE, dtype=np.float64), "dtype"),
        (np.full(INPUT_SHAPE, np.nan, dtype=np.float32), "finite"),
        (np.zeros((181, 3, 72, 144), dtype=np.float32)[:, :, :, ::2], "C-contiguous"),
    ],
)
def test_run_onnx_rejects_invalid_frames_before_session(
    monkeypatch, tmp_path, frames, message
):
    monkeypatch.setattr(
        validator.ort,
        "InferenceSession",
        lambda *_args, **_kwargs: pytest.fail("session must not be created"),
    )
    with pytest.raises(ValueError, match=message):
        validator.run_onnx(tmp_path / "model.onnx", frames)


@pytest.mark.parametrize(
    ("attribute", "value", "message"),
    [
        ("inputs", [], "exactly one input"),
        ("inputs", [_Node("wrong", list(INPUT_SHAPE))], "named 'frames'"),
        ("inputs", [_Node("frames", [None, 3, 72, 72])], "static shape"),
        ("inputs", [_Node("frames", list(INPUT_SHAPE), "tensor(double)")], "float32"),
        ("outputs", [], "exactly one output"),
        ("outputs", [_Node("wrong", list(OUTPUT_SHAPE))], "named 'pulse'"),
        ("outputs", [_Node("pulse", [None, 1])], "static shape"),
        ("outputs", [_Node("pulse", list(OUTPUT_SHAPE), "tensor(double)")], "float32"),
        ("providers", ["CoreMLExecutionProvider"], "CPUExecutionProvider"),
        ("result", np.zeros((180,), dtype=np.float32), "output.*shape"),
        ("result", np.zeros(OUTPUT_SHAPE, dtype=np.float64), "output.*dtype"),
        ("result", np.full(OUTPUT_SHAPE, np.nan, dtype=np.float32), "output.*finite"),
    ],
)
def test_run_onnx_rejects_bad_session_contract(
    monkeypatch, tmp_path, attribute, value, message
):
    session_type = type("Session", (_Session,), {attribute: value})
    monkeypatch.setattr(validator.ort, "InferenceSession", session_type)
    with pytest.raises(ValueError, match=message):
        validator.run_onnx(
            tmp_path / "model.onnx", np.zeros(INPUT_SHAPE, dtype=np.float32)
        )


def _write_validation_fixture(tmp_path, monkeypatch, candidate=None):
    artifact_dir = tmp_path / "artifacts"
    toolbox = tmp_path / "toolbox"
    checkpoint = toolbox / "final_model_release" / "PURE_EfficientPhys.pth"
    artifact_dir.mkdir()
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    official = validator.OFFICIAL_SOURCE_FILES
    for relative in official.values():
        path = toolbox / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative)
    source = np.zeros((180, 72, 72, 3), dtype=np.uint8)
    frames = np.zeros(INPUT_SHAPE, dtype=np.float32)
    pytorch_output = _pulse()
    arrays = {
        "source_rgb_uint8.npy": source,
        "frames_float32.npy": frames,
        "pytorch_pulse_float32.npy": pytorch_output,
    }
    for name, array in arrays.items():
        np.save(artifact_dir / name, array, allow_pickle=False)
    onnx_path = artifact_dir / "efficientphys_pure.onnx"
    onnx_path.write_bytes(b"onnx")
    (artifact_dir / "export_report.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "passed": True,
                "exporter_mode": validator.EXPORTER_MODE,
                "qnn_conversion": "not_run",
                "qnn_risk": validator.QNN_RISK,
                "onnx_sha256": validator.sha256_file(onnx_path),
                "onnx_size_bytes": onnx_path.stat().st_size,
                "onnx_metrics": {
                    "constant_tensor_byte_limit": 216000000,
                    "constant_tensor_bytes": 215315552,
                    "largest_constant_tensors": [],
                    "scatter_nd_nodes": 12,
                },
            }
        )
    )
    monkeypatch.setattr(
        validator, "EXPECTED_CHECKPOINT_SHA256", validator.sha256_file(checkpoint)
    )
    monkeypatch.setattr(
        validator,
        "EXPECTED_RAW_SHA256",
        {key: _raw_sha256(array) for key, array in arrays.items()},
    )
    monkeypatch.setattr(
        validator,
        "EXPECTED_OFFICIAL_SOURCE_SHA256",
        {
            key: validator.sha256_file(toolbox / relative)
            for key, relative in official.items()
        },
        raising=False,
    )
    monkeypatch.setattr(
        validator,
        "_run_onnx_with_metadata",
        lambda *_args: (
            pytorch_output.copy() if candidate is None else candidate,
            {
                "requested_provider": "CPUExecutionProvider",
                "providers": ["CPUExecutionProvider"],
                "inference_seconds": 0.125,
            },
        ),
    )
    return toolbox, checkpoint, artifact_dir


def test_validate_and_publish_writes_complete_portable_manifest(tmp_path, monkeypatch):
    toolbox, checkpoint, artifact_dir = _write_validation_fixture(tmp_path, monkeypatch)
    manifest_path = tmp_path / "model_specs" / "efficientphys_pure.json"

    manifest = validator.validate_and_publish(
        toolbox, checkpoint, artifact_dir, manifest_path
    )

    assert manifest == json.loads(manifest_path.read_text())
    assert manifest["schema_version"] == 1
    assert manifest["model_name"] == "EfficientPhys"
    assert manifest["checkpoint"]["filename"] == checkpoint.name
    assert set(manifest["official_sources"]) == {"model", "trainer", "config"}
    assert manifest["preprocessing"]["frame_rate"] == 30.0
    assert manifest["seed"] == 20260722
    assert manifest["interface"]["input"]["shape"] == list(INPUT_SHAPE)
    assert manifest["exporter_mode"] == validator.EXPORTER_MODE
    assert manifest["onnx"]["opset"] == 17
    assert manifest["onnx_metrics"]["scatter_nd_nodes"] == 12
    assert manifest["runtime"]["providers"] == ["CPUExecutionProvider"]
    assert manifest["validation"]["passed"] is True
    assert manifest["qnn_conversion"] == "not_run"
    assert manifest["qnn_risk"] == validator.QNN_RISK
    for entry in manifest["artifacts"].values():
        assert set(entry) == {
            "file",
            "sha256",
            "size_bytes",
            "shape",
            "dtype",
            "raw_sha256",
        }
    assert "/Users/" not in json.dumps(manifest)
    assert (
        json.loads((artifact_dir / "validation_report.json").read_text())["passed"]
        is True
    )


def test_parity_failure_writes_report_and_preserves_committed_manifest(
    tmp_path, monkeypatch
):
    candidate = _pulse() + np.linspace(0, 0.1, 180, dtype=np.float32).reshape(
        OUTPUT_SHAPE
    )
    toolbox, checkpoint, artifact_dir = _write_validation_fixture(
        tmp_path, monkeypatch, candidate
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("old-manifest")

    with pytest.raises(RuntimeError, match="parity"):
        validator.validate_and_publish(toolbox, checkpoint, artifact_dir, manifest_path)

    assert manifest_path.read_text() == "old-manifest"
    report = json.loads((artifact_dir / "validation_report.json").read_text())
    assert report["passed"] is False
    assert report["validation"]["passed"] is False


def test_validate_rejects_raw_hash_mismatch_before_ort(tmp_path, monkeypatch):
    toolbox, checkpoint, artifact_dir = _write_validation_fixture(tmp_path, monkeypatch)
    expected = dict(validator.EXPECTED_RAW_SHA256)
    expected["frames_float32.npy"] = "0" * 64
    monkeypatch.setattr(validator, "EXPECTED_RAW_SHA256", expected)
    monkeypatch.setattr(
        validator,
        "_run_onnx_with_metadata",
        lambda *_args: pytest.fail("ORT must run after artifact validation"),
    )

    with pytest.raises(ValueError, match="frames_float32.npy raw SHA-256 mismatch"):
        validator.validate_and_publish(
            toolbox, checkpoint, artifact_dir, tmp_path / "manifest"
        )


def test_validate_rejects_modified_official_source_before_ort_and_preserves_manifest(
    tmp_path, monkeypatch
):
    toolbox, checkpoint, artifact_dir = _write_validation_fixture(tmp_path, monkeypatch)
    (toolbox / validator.OFFICIAL_SOURCE_FILES["model"]).write_text("modified")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("known-good")
    monkeypatch.setattr(
        validator,
        "_run_onnx_with_metadata",
        lambda *_args: pytest.fail("ORT must run after official source validation"),
    )

    with pytest.raises(ValueError, match="official model source SHA-256 mismatch"):
        validator.validate_and_publish(toolbox, checkpoint, artifact_dir, manifest_path)

    assert manifest_path.read_text() == "known-good"


@pytest.mark.parametrize("tamper_report", [True, False])
def test_validate_rejects_export_report_not_bound_to_current_onnx_before_ort(
    tmp_path, monkeypatch, tamper_report
):
    toolbox, checkpoint, artifact_dir = _write_validation_fixture(tmp_path, monkeypatch)
    if tamper_report:
        report_path = artifact_dir / "export_report.json"
        report = json.loads(report_path.read_text())
        report["onnx_sha256"] = "0" * 64
        report_path.write_text(json.dumps(report))
        message = "export report ONNX SHA-256 mismatch"
    else:
        (artifact_dir / "efficientphys_pure.onnx").write_bytes(b"tampered-onnx")
        message = "export report ONNX"
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("known-good")
    monkeypatch.setattr(
        validator,
        "_run_onnx_with_metadata",
        lambda *_args: pytest.fail("ORT must run after report binding validation"),
    )

    with pytest.raises(ValueError, match=message):
        validator.validate_and_publish(toolbox, checkpoint, artifact_dir, manifest_path)

    assert manifest_path.read_text() == "known-good"


def test_cli_failure_from_outside_repo_writes_report_without_changing_manifest(
    tmp_path,
):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools" / "model_export" / "validate_efficientphys.py"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    manifest = tmp_path / "manifest.json"
    manifest.write_text("known-good")

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--toolbox",
            str(tmp_path / "toolbox"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--artifact-dir",
            str(artifact_dir),
            "--manifest",
            str(manifest),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert manifest.read_text() == "known-good"
    report = json.loads((artifact_dir / "validation_report.json").read_text())
    assert report["schema_version"] == 1
    assert report["passed"] is False
    assert report["error"]


def _integration_resource(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        pytest.skip(f"{variable} is required for the validation integration test")
    return Path(value)


@pytest.mark.integration
def test_real_cli_validates_with_onnxruntime_from_outside_repo(tmp_path):
    toolbox = _integration_resource("RPPG_TOOLBOX_PATH")
    checkpoint = _integration_resource("RPPG_EFFICIENTPHYS_CHECKPOINT")
    artifact_dir = _integration_resource("RPPG_MODEL_ARTIFACT_DIR")
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools" / "model_export" / "validate_efficientphys.py"
    manifest_path = tmp_path / "efficientphys_pure.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--toolbox",
            str(toolbox),
            "--checkpoint",
            str(checkpoint),
            "--artifact-dir",
            str(artifact_dir),
            "--manifest",
            str(manifest_path),
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads(manifest_path.read_text())
    assert manifest["validation"]["passed"] is True
    assert manifest["runtime"]["providers"] == ["CPUExecutionProvider"]
    assert manifest["qnn_conversion"] == "not_run"
    assert "/Users/" not in json.dumps(manifest)
