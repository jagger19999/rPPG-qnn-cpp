"""Export the fixed EfficientPhys reference model and PyTorch output."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import threading
from typing import Any

import numpy as np
import onnx
from onnx import TensorProto
import torch
from torch.onnx import symbolic_helper


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model_export.efficientphys_reference import load_official_model


MODEL_INPUT_SHAPE = (181, 3, 72, 72)
MODEL_OUTPUT_SHAPE = (180, 1)
ONNX_FILENAME = "efficientphys_pure.onnx"
PYTORCH_OUTPUT_FILENAME = "pytorch_pulse_float32.npy"
REPORT_FILENAME = "export_report.json"
EXPORTER_MODE = (
    "legacy_dynamo_false_with_explicit_aten_diff_slice_sub_lowering"
)
_EXPORT_LOCK = threading.Lock()


@dataclass(frozen=True)
class ExportPaths:
    onnx: Path
    pytorch_output: Path


def _validate_frames(frames: np.ndarray) -> np.ndarray:
    if not isinstance(frames, np.ndarray):
        raise ValueError(
            "frames must be a NumPy ndarray with shape "
            f"{MODEL_INPUT_SHAPE} and dtype float32"
        )
    if frames.shape != MODEL_INPUT_SHAPE:
        raise ValueError(
            f"frames shape must be {MODEL_INPUT_SHAPE}; got {frames.shape}"
        )
    if frames.dtype != np.dtype(np.float32):
        raise ValueError(f"frames dtype must be exactly float32; got {frames.dtype}")
    if not np.isfinite(frames).all():
        raise ValueError("frames must contain only finite values")
    try:
        contiguous = np.ascontiguousarray(frames)
    except Exception as error:
        raise ValueError("frames must be convertible to a C-contiguous array") from error
    if not contiguous.flags.c_contiguous:
        raise ValueError("frames must be convertible to a C-contiguous array")
    return contiguous


def _save_numpy_temporary(temporary: Path, array: np.ndarray) -> None:
    try:
        with temporary.open("wb") as output:
            np.save(output, array, allow_pickle=False)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


@symbolic_helper.parse_args("v", "i", "i", "v", "v")
def _aten_diff_symbolic(graph, inputs, n, dim, prepend, append):
    """Lower only the exact ``torch.diff`` form used by official EfficientPhys."""
    if n != 1:
        raise RuntimeError("aten::diff legacy lowering supports only n=1")
    if dim != 0:
        raise RuntimeError("aten::diff legacy lowering supports only dim=0")
    if not symbolic_helper._is_none(prepend):
        raise RuntimeError("aten::diff legacy lowering supports only prepend=None")
    if not symbolic_helper._is_none(append):
        raise RuntimeError("aten::diff legacy lowering supports only append=None")

    def int64_constant(values):
        return graph.op(
            "Constant",
            value_t=torch.tensor(values, dtype=torch.int64),
        )

    axes = int64_constant([0])
    steps = int64_constant([1])
    tail = graph.op(
        "Slice",
        inputs,
        int64_constant([1]),
        int64_constant([sys.maxsize]),
        axes,
        steps,
    )
    head = graph.op(
        "Slice",
        inputs,
        int64_constant([0]),
        int64_constant([-1]),
        axes,
        steps,
    )
    return graph.op("Sub", tail, head)


def _static_shape(value_info: onnx.ValueInfoProto) -> list[int]:
    tensor_type = value_info.type.tensor_type
    dimensions = tensor_type.shape.dim
    if any(dimension.HasField("dim_param") for dimension in dimensions):
        raise ValueError(f"ONNX value {value_info.name!r} must have a static shape")
    if any(not dimension.HasField("dim_value") for dimension in dimensions):
        raise ValueError(f"ONNX value {value_info.name!r} must have a static shape")
    return [dimension.dim_value for dimension in dimensions]


def _validate_onnx(path: Path) -> None:
    model = onnx.load(path)
    onnx.checker.check_model(model)

    custom_node_domains = sorted({node.domain for node in model.graph.node if node.domain})
    custom_opset_domains = sorted(
        {item.domain for item in model.opset_import if item.domain}
    )
    if custom_node_domains or custom_opset_domains:
        raise ValueError(
            "ONNX graph must not contain a custom domain; "
            f"node domains={custom_node_domains}, opset domains={custom_opset_domains}"
        )

    if len(model.graph.input) != 1:
        raise ValueError("ONNX graph must have exactly 1 input")
    if len(model.graph.output) != 1:
        raise ValueError("ONNX graph must have exactly 1 output")

    graph_input = model.graph.input[0]
    graph_output = model.graph.output[0]
    if graph_input.name != "frames":
        raise ValueError(
            f"ONNX input name must be 'frames'; got {graph_input.name!r}"
        )
    if graph_output.name != "pulse":
        raise ValueError(
            f"ONNX output name must be 'pulse'; got {graph_output.name!r}"
        )
    if _static_shape(graph_input) != list(MODEL_INPUT_SHAPE):
        raise ValueError(
            f"ONNX input shape must be {list(MODEL_INPUT_SHAPE)}; "
            f"got {_static_shape(graph_input)}"
        )
    if _static_shape(graph_output) != list(MODEL_OUTPUT_SHAPE):
        raise ValueError(
            f"ONNX output shape must be {list(MODEL_OUTPUT_SHAPE)}; "
            f"got {_static_shape(graph_output)}"
        )
    if graph_input.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("ONNX input dtype must be float32")
    if graph_output.type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("ONNX output dtype must be float32")
    if [(item.domain, item.version) for item in model.opset_import] != [("", 17)]:
        raise ValueError("ONNX graph must use exactly the default-domain opset 17")


def export_reference(
    toolbox: Path,
    checkpoint: Path,
    frames: np.ndarray,
    artifact_dir: Path,
) -> ExportPaths:
    """Export one fixed-shape ONNX model and its PyTorch reference output."""
    contiguous_frames = _validate_frames(frames)
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model = load_official_model(Path(toolbox), Path(checkpoint))
    input_tensor = torch.from_numpy(contiguous_frames)
    with torch.inference_mode():
        output_tensor = model(input_tensor)
    output = output_tensor.detach().cpu().numpy()
    if output.shape != MODEL_OUTPUT_SHAPE:
        raise ValueError(
            f"PyTorch output shape must be {MODEL_OUTPUT_SHAPE}; got {output.shape}"
        )
    if output.dtype != np.dtype(np.float32):
        raise ValueError(
            f"PyTorch output dtype must be exactly float32; got {output.dtype}"
        )
    if not np.isfinite(output).all():
        raise ValueError("PyTorch output must contain only finite values")

    pytorch_output_path = artifact_dir / PYTORCH_OUTPUT_FILENAME
    onnx_path = artifact_dir / ONNX_FILENAME
    temporary_output_path = artifact_dir / f"{PYTORCH_OUTPUT_FILENAME}.tmp"
    temporary_onnx_path = artifact_dir / f"{ONNX_FILENAME}.tmp"
    try:
        _save_numpy_temporary(temporary_output_path, output)
        with _EXPORT_LOCK:
            registered = False
            try:
                torch.onnx.register_custom_op_symbolic(
                    "aten::diff",
                    _aten_diff_symbolic,
                    17,
                )
                registered = True
                torch.onnx.export(
                    model,
                    input_tensor,
                    temporary_onnx_path,
                    input_names=["frames"],
                    output_names=["pulse"],
                    opset_version=17,
                    do_constant_folding=True,
                    dynamo=False,
                )
            finally:
                if registered:
                    torch.onnx.unregister_custom_op_symbolic("aten::diff", 17)
        _validate_onnx(temporary_onnx_path)
        os.replace(temporary_output_path, pytorch_output_path)
        os.replace(temporary_onnx_path, onnx_path)
    except BaseException:
        temporary_output_path.unlink(missing_ok=True)
        temporary_onnx_path.unlink(missing_ok=True)
        raise

    return ExportPaths(onnx=onnx_path, pytorch_output=pytorch_output_path)


def _atomic_write_report(path: Path, report: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output:
            json.dump(report, output, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toolbox", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.artifact_dir / REPORT_FILENAME
    try:
        frames = np.load(
            args.artifact_dir / "frames_float32.npy",
            allow_pickle=False,
        )
        paths = export_reference(
            args.toolbox,
            args.checkpoint,
            frames,
            args.artifact_dir,
        )
    except Exception as error:
        _atomic_write_report(
            report_path,
            {
                "schema_version": 1,
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
                "exporter_mode": EXPORTER_MODE,
            },
        )
        raise

    _atomic_write_report(
        report_path,
        {
            "schema_version": 1,
            "passed": True,
            "exporter_mode": EXPORTER_MODE,
        },
    )
    print(paths.onnx)
    print(paths.pytorch_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
