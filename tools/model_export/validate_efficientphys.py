"""Validate fixed EfficientPhys ONNX parity and publish its portable manifest."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import sys
import tempfile
import time
from typing import Any

import numpy as np
import onnxruntime as ort


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model_export.efficientphys_reference import (  # noqa: E402
    EXPECTED_CHECKPOINT_SHA256,
    FRAME_DEPTH,
    FRAME_RATE,
    sha256_file,
)
from tools.model_export.export_efficientphys import (  # noqa: E402
    EXPORTER_MODE,
    MAX_CONSTANT_TENSOR_BYTES,
    QNN_RISK,
)
from tools.model_export.generate_test_vector import GENERATOR_SEED  # noqa: E402


MODEL_INPUT_SHAPE = (181, 3, 72, 72)
MODEL_OUTPUT_SHAPE = (180, 1)
SOURCE_SHAPE = (180, 72, 72, 3)
ONNX_FILENAME = "efficientphys_pure.onnx"
VALIDATION_REPORT_FILENAME = "validation_report.json"
EXPECTED_RAW_SHA256 = {
    "source_rgb_uint8.npy": (
        "dca274b1e509762f17a2dff152fa0fc3fa87c0c6f2726db284792d3494350b46"
    ),
    "frames_float32.npy": (
        "b685ed04d5b6a96bbdd2f97fed6cc04a8a2eb35c38b7ebc820eb1a37af2d2f40"
    ),
    "pytorch_pulse_float32.npy": (
        "5c7d6202e56f02d4727571afbae3048368566e6d8b549c5d55544408282f5a56"
    ),
}
OFFICIAL_SOURCE_FILES = {
    "model": "neural_methods/model/EfficientPhys.py",
    "trainer": "neural_methods/trainer/EfficientPhysTrainer.py",
    "config": "configs/infer_configs/PURE_UBFC-rPPG_EFFICIENTPHYS.yaml",
}
EXPECTED_OFFICIAL_SOURCE_SHA256 = {
    "model": "384b01b5c99d6d280b5ddb9e2358631f89747e40e3181b6a174e26d6e74798c9",
    "trainer": "21336fa95a36524eb754710ae7837ba3fdc37e9b370c3be67456db3024e3c5e8",
    "config": "579105d1799307b8e11b56d68b53f401f1f32ff59c65be6fd5f1fcafc6ece370",
}
THRESHOLDS = {
    "max_abs_error": 1e-4,
    "mean_abs_error": 1e-5,
    "pearson": 0.99999,
    "bpm_error": 0.1,
}


class ParityError(RuntimeError):
    """Raised after a detailed failed-parity report has been published."""


def _raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def fft_bpm(waveform: np.ndarray, fps: float) -> float:
    """Return the dominant 0.7--3.0 Hz frequency as beats per minute."""
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError("fps must be finite and positive")
    signal = np.asarray(waveform).reshape(-1).astype(np.float64)
    if signal.size == 0 or not np.isfinite(signal).all():
        raise ValueError("cannot derive finite positive band power")
    signal -= signal.mean(dtype=np.float64)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / float(fps))
    power = np.abs(np.fft.rfft(signal)) ** 2
    mask = (frequencies >= 0.7) & (frequencies <= 3.0)
    band_power = power[mask]
    if (
        band_power.size == 0
        or not np.isfinite(band_power).all()
        or not np.any(band_power > 0.0)
    ):
        raise ValueError("cannot derive finite positive band power")
    return float(frequencies[mask][int(np.argmax(band_power))] * 60.0)


def _validate_output(name: str, value: np.ndarray) -> None:
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} must be an ndarray with shape {MODEL_OUTPUT_SHAPE}")
    if value.shape != MODEL_OUTPUT_SHAPE:
        raise ValueError(
            f"{name} must have shape {MODEL_OUTPUT_SHAPE}; got {value.shape}"
        )
    if value.dtype != np.dtype(np.float32):
        raise ValueError(f"{name} dtype must be exactly float32; got {value.dtype}")
    if not np.isfinite(value).all():
        raise ValueError(f"{name} must contain only finite values")
    standard_deviation = float(value.astype(np.float64).std(ddof=0))
    if not np.isfinite(standard_deviation) or standard_deviation <= 0.0:
        raise ValueError(f"{name} standard deviation must be finite and positive")


def compare_outputs(
    reference: np.ndarray, candidate: np.ndarray, fps: float
) -> dict[str, Any]:
    """Evaluate the fixed numerical and waveform-parity gates."""
    _validate_output("reference output", reference)
    _validate_output("candidate output", candidate)
    reference64 = reference.reshape(-1).astype(np.float64)
    candidate64 = candidate.reshape(-1).astype(np.float64)
    difference = np.abs(reference64 - candidate64)
    pearson = float(np.corrcoef(reference64, candidate64)[0, 1])
    if not np.isfinite(pearson):
        raise ValueError("Pearson correlation must be finite")
    bpm_error = abs(fft_bpm(reference64, fps) - fft_bpm(candidate64, fps))
    maximum = float(difference.max())
    mean = float(difference.mean())
    passed = (
        maximum <= THRESHOLDS["max_abs_error"]
        and mean <= THRESHOLDS["mean_abs_error"]
        and pearson >= THRESHOLDS["pearson"]
        and bpm_error <= THRESHOLDS["bpm_error"]
    )
    return {
        "max_abs_error": maximum,
        "mean_abs_error": mean,
        "pearson": pearson,
        "bpm_error": bpm_error,
        "thresholds": dict(THRESHOLDS),
        "passed": bool(passed),
    }


def publish_json_atomic(
    path: Path, payload: dict[str, Any], *, fsync: bool = False
) -> None:
    """Atomically publish JSON using a unique same-directory temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            if fsync:
                os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _publish_numpy_atomic(
    path: Path, array: np.ndarray, *, fsync: bool = False
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.save(output, array, allow_pickle=False)
            output.flush()
            if fsync:
                os.fsync(output.fileno())
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_node(node: Any, *, kind: str, name: str, shape: tuple[int, ...]) -> None:
    if node.name != name:
        raise ValueError(f"ONNX {kind} must be named {name!r}; got {node.name!r}")
    if list(node.shape) != list(shape) or any(
        not isinstance(value, int) for value in node.shape
    ):
        raise ValueError(
            f"ONNX {kind} must have static shape {list(shape)}; got {node.shape}"
        )
    if node.type != "tensor(float)":
        raise ValueError(f"ONNX {kind} dtype must be float32; got {node.type!r}")


def _validate_frames(frames: np.ndarray) -> None:
    if not isinstance(frames, np.ndarray):
        raise TypeError("frames must be a NumPy ndarray")
    if frames.shape != MODEL_INPUT_SHAPE:
        raise ValueError(
            f"frames shape must be {MODEL_INPUT_SHAPE}; got {frames.shape}"
        )
    if frames.dtype != np.dtype(np.float32):
        raise ValueError(f"frames dtype must be exactly float32; got {frames.dtype}")
    if not np.isfinite(frames).all():
        raise ValueError("frames must contain only finite values")
    if not frames.flags.c_contiguous:
        raise ValueError("frames must be C-contiguous")


def _run_onnx_with_metadata(
    onnx_path: Path, frames: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    _validate_frames(frames)
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    if len(inputs) != 1:
        raise ValueError("ONNX graph must expose exactly one input")
    if len(outputs) != 1:
        raise ValueError("ONNX graph must expose exactly one output")
    _validate_node(inputs[0], kind="input", name="frames", shape=MODEL_INPUT_SHAPE)
    _validate_node(outputs[0], kind="output", name="pulse", shape=MODEL_OUTPUT_SHAPE)
    providers = list(session.get_providers())
    if "CPUExecutionProvider" not in providers:
        raise ValueError("ONNX Runtime session must use CPUExecutionProvider")
    started = time.perf_counter()
    output = np.asarray(session.run(["pulse"], {"frames": frames})[0])
    elapsed = time.perf_counter() - started
    if output.shape != MODEL_OUTPUT_SHAPE:
        raise ValueError(
            f"ONNX output shape must be {MODEL_OUTPUT_SHAPE}; got {output.shape}"
        )
    if output.dtype != np.dtype(np.float32):
        raise ValueError(
            f"ONNX output dtype must be exactly float32; got {output.dtype}"
        )
    if not np.isfinite(output).all():
        raise ValueError("ONNX output must contain only finite values")
    return output, {
        "requested_provider": "CPUExecutionProvider",
        "providers": providers,
        "inference_seconds": float(elapsed),
    }


def run_onnx(onnx_path: Path, frames: np.ndarray) -> np.ndarray:
    return _run_onnx_with_metadata(onnx_path, frames)[0]


def package_versions() -> dict[str, str]:
    return {
        name: importlib.metadata.version(name)
        for name in ("numpy", "torch", "onnx", "onnxruntime")
    }


def _environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": package_versions(),
    }


def _load_array(path: Path, shape: tuple[int, ...], dtype: np.dtype[Any]) -> np.ndarray:
    array = np.load(path, allow_pickle=False)
    if not isinstance(array, np.ndarray):
        raise TypeError(f"{path.name} must contain a NumPy ndarray")
    if array.shape != shape:
        raise ValueError(f"{path.name} shape must be {shape}; got {array.shape}")
    if array.dtype != np.dtype(dtype):
        raise ValueError(
            f"{path.name} dtype must be exactly {np.dtype(dtype)}; got {array.dtype}"
        )
    if not np.isfinite(array).all():
        raise ValueError(f"{path.name} must contain only finite values")
    expected_raw_sha256 = EXPECTED_RAW_SHA256[path.name]
    actual_raw_sha256 = _raw_sha256(array)
    if actual_raw_sha256 != expected_raw_sha256:
        raise ValueError(
            f"{path.name} raw SHA-256 mismatch: expected {expected_raw_sha256}, "
            f"got {actual_raw_sha256}"
        )
    return array


def _artifact_entry(path: Path, array: np.ndarray) -> dict[str, Any]:
    return {
        "file": path.name,
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "raw_sha256": _raw_sha256(array),
    }


def _load_export_report(path: Path, onnx_path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        report = json.load(source)
    if report.get("schema_version") != 1 or report.get("passed") is not True:
        raise ValueError("export report must be a passed schema-version-1 report")
    if report.get("exporter_mode") != EXPORTER_MODE:
        raise ValueError("export report exporter_mode mismatch")
    if report.get("qnn_conversion") != "not_run" or report.get("qnn_risk") != QNN_RISK:
        raise ValueError("export report QNN status mismatch")
    actual_onnx_sha256 = sha256_file(onnx_path)
    if report.get("onnx_sha256") != actual_onnx_sha256:
        raise ValueError(
            "export report ONNX SHA-256 mismatch: "
            f"expected current file {actual_onnx_sha256}, "
            f"got {report.get('onnx_sha256')!r}"
        )
    actual_onnx_size = onnx_path.stat().st_size
    if report.get("onnx_size_bytes") != actual_onnx_size:
        raise ValueError(
            "export report ONNX size mismatch: "
            f"expected current file {actual_onnx_size}, "
            f"got {report.get('onnx_size_bytes')!r}"
        )
    metrics = report.get("onnx_metrics")
    if not isinstance(metrics, dict):
        raise ValueError("export report must contain onnx_metrics")
    required = {
        "constant_tensor_byte_limit",
        "constant_tensor_bytes",
        "largest_constant_tensors",
        "scatter_nd_nodes",
    }
    if not required.issubset(metrics):
        raise ValueError("export report onnx_metrics are incomplete")
    if metrics["constant_tensor_byte_limit"] != MAX_CONSTANT_TENSOR_BYTES:
        raise ValueError("ONNX constant tensor byte limit mismatch")
    if (
        not isinstance(metrics["constant_tensor_bytes"], int)
        or metrics["constant_tensor_bytes"] < 0
        or metrics["constant_tensor_bytes"] > metrics["constant_tensor_byte_limit"]
    ):
        raise ValueError("ONNX constant tensor byte gate failed")
    if metrics["scatter_nd_nodes"] != 12:
        raise ValueError("ONNX ScatterND node gate failed")
    if not isinstance(metrics["largest_constant_tensors"], list):
        raise ValueError("ONNX largest constant tensors must be a list")
    return report


def validate_and_publish(
    toolbox: Path, checkpoint: Path, artifact_dir: Path, manifest_path: Path
) -> dict[str, Any]:
    """Validate all pinned inputs, execute ORT, and publish a passed manifest."""
    toolbox = Path(toolbox)
    checkpoint = Path(checkpoint)
    artifact_dir = Path(artifact_dir)
    manifest_path = Path(manifest_path)
    source_path = artifact_dir / "source_rgb_uint8.npy"
    frames_path = artifact_dir / "frames_float32.npy"
    pytorch_path = artifact_dir / "pytorch_pulse_float32.npy"
    onnx_path = artifact_dir / ONNX_FILENAME
    export_report_path = artifact_dir / "export_report.json"

    source = _load_array(source_path, SOURCE_SHAPE, np.uint8)
    frames = _load_array(frames_path, MODEL_INPUT_SHAPE, np.float32)
    pytorch_output = _load_array(pytorch_path, MODEL_OUTPUT_SHAPE, np.float32)
    checkpoint_sha256 = sha256_file(checkpoint)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )
    if not onnx_path.is_file():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path.name}")
    export_report = _load_export_report(export_report_path, onnx_path)

    official_sources = {}
    for name, relative in OFFICIAL_SOURCE_FILES.items():
        official_path = toolbox / relative
        official_sha256 = sha256_file(official_path)
        expected_sha256 = EXPECTED_OFFICIAL_SOURCE_SHA256[name]
        if official_sha256 != expected_sha256:
            raise ValueError(
                f"official {name} source SHA-256 mismatch: "
                f"expected {expected_sha256}, got {official_sha256}"
            )
        official_sources[name] = {
            "file": relative,
            "sha256": official_sha256,
        }
    onnx_output, runtime = _run_onnx_with_metadata(onnx_path, frames)
    onnx_output_path = artifact_dir / "onnx_pulse_float32.npy"
    _publish_numpy_atomic(onnx_output_path, onnx_output)
    validation = compare_outputs(pytorch_output, onnx_output, FRAME_RATE)
    environment = _environment()
    report = {
        "schema_version": 1,
        "passed": validation["passed"],
        "onnx": {"file": onnx_path.name, "sha256": sha256_file(onnx_path)},
        "runtime": runtime,
        "validation": validation,
        "environment": environment,
    }
    publish_json_atomic(artifact_dir / VALIDATION_REPORT_FILENAME, report)
    if not validation["passed"]:
        raise ParityError("PyTorch and ONNX outputs failed parity gates")

    artifacts = {
        "source": _artifact_entry(source_path, source),
        "frames": _artifact_entry(frames_path, frames),
        "pytorch_output": _artifact_entry(pytorch_path, pytorch_output),
        "onnx_output": _artifact_entry(onnx_output_path, onnx_output),
    }
    manifest = {
        "schema_version": 1,
        "model_name": "EfficientPhys",
        "checkpoint": {"filename": checkpoint.name, "sha256": checkpoint_sha256},
        "official_sources": official_sources,
        "preprocessing": {
            "source_shape": list(SOURCE_SHAPE),
            "source_dtype": "uint8",
            "source_layout": "THWC",
            "color_order": "RGB",
            "standardization": "one population mean/std over complete source window",
            "append_last_frame": True,
            "frame_rate": FRAME_RATE,
        },
        "seed": GENERATOR_SEED,
        "test_vector": {"algorithm_version": 1, "seed": GENERATOR_SEED},
        "interface": {
            "input": {
                "name": "frames",
                "shape": list(MODEL_INPUT_SHAPE),
                "dtype": "float32",
                "layout": "TCHW",
            },
            "output": {
                "name": "pulse",
                "shape": list(MODEL_OUTPUT_SHAPE),
                "dtype": "float32",
            },
            "frame_depth": FRAME_DEPTH,
        },
        "exporter_mode": EXPORTER_MODE,
        "onnx": {
            "file": onnx_path.name,
            "sha256": sha256_file(onnx_path),
            "size_bytes": onnx_path.stat().st_size,
            "opset": 17,
        },
        "onnx_metrics": export_report["onnx_metrics"],
        "artifacts": artifacts,
        "environment": environment,
        "runtime": {
            "requested_provider": runtime["requested_provider"],
            "providers": runtime["providers"],
        },
        "validation": validation,
        "qnn_conversion": "not_run",
        "qnn_risk": QNN_RISK,
    }
    serialized = json.dumps(manifest)
    if "/Users/" in serialized:
        raise ValueError("manifest must not contain absolute private paths")
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
        validate_and_publish(
            args.toolbox, args.checkpoint, args.artifact_dir, args.manifest
        )
    except ParityError:
        raise
    except Exception as error:
        publish_json_atomic(
            args.artifact_dir / VALIDATION_REPORT_FILENAME,
            {
                "schema_version": 1,
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            },
        )
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
