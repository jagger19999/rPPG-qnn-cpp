from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper
import pytest
import torch
from torch.onnx._internal.torchscript_exporter import registration

from tools.model_export import export_efficientphys as exporter


INPUT_SHAPE = (181, 3, 72, 72)
OUTPUT_SHAPE = (180, 1)


class _FakeModel(torch.nn.Module):
    def forward(self, frames):
        return frames[:180, :1, :1, :1].reshape(180, 1)


class _DiffModel(torch.nn.Module):
    def forward(self, frames):
        return torch.diff(frames[:, 0, 0, 0], dim=0).reshape(180, 1)


def _valid_frames() -> np.ndarray:
    return np.zeros(INPUT_SHAPE, dtype=np.float32)


def _write_valid_onnx(path: Path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["frames"], ["pulse"])],
        "fixed-efficientphys",
        [helper.make_tensor_value_info("frames", TensorProto.FLOAT, INPUT_SHAPE)],
        [helper.make_tensor_value_info("pulse", TensorProto.FLOAT, OUTPUT_SHAPE)],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), path)


def _install_fake_export(monkeypatch, calls=None):
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _FakeModel())

    def fake_export(model, frames, destination, **kwargs):
        if calls is not None:
            calls.append((model, frames, Path(destination), kwargs))
        _write_valid_onnx(Path(destination))

    monkeypatch.setattr(torch.onnx, "export", fake_export)


def test_export_paths_is_frozen_dataclass():
    fields = [field.name for field in dataclasses.fields(exporter.ExportPaths)]
    paths = exporter.ExportPaths(Path("model.onnx"), Path("pulse.npy"))

    assert fields == ["onnx", "pytorch_output"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        paths.onnx = Path("changed.onnx")


@pytest.mark.parametrize(
    ("frames", "message"),
    [
        ([0.0], "NumPy ndarray"),
        (np.zeros((180, 3, 72, 72), dtype=np.float32), "shape"),
        (np.zeros(INPUT_SHAPE, dtype=np.float64), "dtype"),
        (
            np.full(INPUT_SHAPE, np.nan, dtype=np.float32),
            "finite",
        ),
    ],
)
def test_export_rejects_invalid_frames_before_loading_model(
    tmp_path, monkeypatch, frames, message
):
    def unexpected_load(*_args, **_kwargs):
        raise AssertionError("model loading must happen after frame validation")

    monkeypatch.setattr(exporter, "load_official_model", unexpected_load)

    with pytest.raises(ValueError, match=message):
        exporter.export_reference(tmp_path, tmp_path / "checkpoint", frames, tmp_path)


def test_export_makes_noncontiguous_frames_contiguous_and_uses_fixed_onnx_contract(
    tmp_path, monkeypatch
):
    calls = []
    _install_fake_export(monkeypatch, calls)
    frames = np.zeros((181, 3, 72, 144), dtype=np.float32)[:, :, :, ::2]
    assert frames.shape == INPUT_SHAPE
    assert not frames.flags.c_contiguous

    paths = exporter.export_reference(
        tmp_path / "toolbox", tmp_path / "checkpoint", frames, tmp_path / "nested"
    )

    assert paths == exporter.ExportPaths(
        tmp_path / "nested" / "efficientphys_pure.onnx",
        tmp_path / "nested" / "pytorch_pulse_float32.npy",
    )
    assert len(calls) == 1
    _model, tensor, destination, kwargs = calls[0]
    assert tensor.is_contiguous()
    assert destination.name == "efficientphys_pure.onnx"
    assert destination.parent.parent == tmp_path / "nested"
    assert destination.parent.name.startswith(".efficientphys-export-")
    assert kwargs == {
        "input_names": ["frames"],
        "output_names": ["pulse"],
        "opset_version": 17,
        "do_constant_folding": True,
        "dynamo": False,
    }
    output = np.load(paths.pytorch_output, allow_pickle=False)
    assert output.shape == OUTPUT_SHAPE
    assert output.dtype == np.float32
    assert np.isfinite(output).all()


def test_real_legacy_export_registers_limited_diff_lowering_and_unregisters_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _DiffModel())
    events = []
    real_register = torch.onnx.register_custom_op_symbolic
    real_unregister = torch.onnx.unregister_custom_op_symbolic

    def recording_register(name, function, opset):
        events.append(("register", name, function, opset))
        return real_register(name, function, opset)

    def recording_unregister(name, opset):
        events.append(("unregister", name, opset))
        return real_unregister(name, opset)

    monkeypatch.setattr(torch.onnx, "register_custom_op_symbolic", recording_register)
    monkeypatch.setattr(torch.onnx, "unregister_custom_op_symbolic", recording_unregister)

    paths = exporter.export_reference(
        tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path
    )

    assert events[0][:2] == ("register", "aten::diff")
    assert events[0][3] == 17
    assert events[-1] == ("unregister", "aten::diff", 17)
    model = onnx.load(paths.onnx)
    operator_types = [node.op_type for node in model.graph.node]
    assert operator_types.count("Slice") >= 2
    assert "Sub" in operator_types
    assert all(node.domain == "" for node in model.graph.node)
    assert [(item.domain, item.version) for item in model.opset_import] == [("", 17)]


def test_export_refuses_preexisting_diff_override_without_altering_it(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _DiffModel())
    export_called = False
    sentinel_called = False
    real_export = torch.onnx.export

    def unexpected_export(*_args, **_kwargs):
        nonlocal export_called
        export_called = True
        raise AssertionError("export must not run when aten::diff is already overridden")

    def sentinel(graph, inputs, _n, _dim, _prepend, _append):
        nonlocal sentinel_called
        sentinel_called = True
        return graph.op("Sub", inputs, inputs)

    torch.onnx.register_custom_op_symbolic("aten::diff", sentinel, 17)
    monkeypatch.setattr(torch.onnx, "export", unexpected_export)
    try:
        with pytest.raises(RuntimeError, match="pre-existing.*aten::diff"):
            exporter.export_reference(
                tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path
            )

        group = registration.registry.get_function_group("aten::diff")
        assert group is not None
        assert group.get(17) is sentinel
        assert not export_called

        tiny_path = tmp_path / "sentinel.onnx"
        real_export(
            _DiffModel(),
            torch.from_numpy(_valid_frames()),
            tiny_path,
            opset_version=17,
            dynamo=False,
        )
        assert sentinel_called
        onnx.checker.check_model(onnx.load(tiny_path))
    finally:
        torch.onnx.unregister_custom_op_symbolic("aten::diff", 17)

    group = registration.registry.get_function_group("aten::diff")
    assert group is None or group.get(17) is None


@pytest.mark.parametrize(
    ("n", "dim", "prepend", "append", "message"),
    [
        (2, 0, None, None, "n=1"),
        (1, 1, None, None, "dim=0"),
        (1, 0, object(), None, "prepend=None"),
        (1, 0, None, object(), "append=None"),
    ],
)
def test_diff_symbolic_rejects_every_nonapproved_parameter(
    n, dim, prepend, append, message
):
    with pytest.raises(RuntimeError, match=message):
        exporter._aten_diff_symbolic(object(), object(), n, dim, prepend, append)


def test_pytorch_output_write_failure_leaves_no_partial_final_or_temp(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _FakeModel())

    def failing_save(output, *_args, **_kwargs):
        output.write(b"partial")
        output.flush()
        raise OSError("disk full")

    monkeypatch.setattr(np, "save", failing_save)

    with pytest.raises(OSError, match="disk full"):
        exporter.export_reference(tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path)

    assert not (tmp_path / "pytorch_pulse_float32.npy").exists()
    assert not list(tmp_path.glob(".efficientphys-export-*"))


def test_onnx_export_failure_cleans_temp_and_preserves_existing_final(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _FakeModel())
    final_onnx = tmp_path / "efficientphys_pure.onnx"
    final_output = tmp_path / "pytorch_pulse_float32.npy"
    final_onnx.write_bytes(b"known-good-onnx")
    final_output.write_bytes(b"known-good-output")
    events = []
    monkeypatch.setattr(
        torch.onnx,
        "register_custom_op_symbolic",
        lambda name, function, opset: events.append(("register", name, opset)),
    )
    monkeypatch.setattr(
        torch.onnx,
        "unregister_custom_op_symbolic",
        lambda name, opset: events.append(("unregister", name, opset)),
    )

    def failing_export(_model, _frames, destination, **_kwargs):
        Path(destination).write_bytes(b"partial")
        raise RuntimeError("export failed")

    monkeypatch.setattr(torch.onnx, "export", failing_export)

    with pytest.raises(RuntimeError, match="export failed"):
        exporter.export_reference(tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path)

    assert events == [
        ("register", "aten::diff", 17),
        ("unregister", "aten::diff", 17),
    ]
    assert final_onnx.read_bytes() == b"known-good-onnx"
    assert final_output.read_bytes() == b"known-good-output"
    assert not list(tmp_path.glob(".efficientphys-export-*"))


def test_invalid_exported_onnx_cleans_temp_and_preserves_existing_final(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _FakeModel())
    final = tmp_path / "efficientphys_pure.onnx"
    final.write_bytes(b"known-good")

    def wrong_interface(_model, _frames, destination, **_kwargs):
        graph = helper.make_graph(
            [helper.make_node("Identity", ["wrong"], ["pulse"])],
            "wrong-interface",
            [helper.make_tensor_value_info("wrong", TensorProto.FLOAT, INPUT_SHAPE)],
            [helper.make_tensor_value_info("pulse", TensorProto.FLOAT, INPUT_SHAPE)],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
            destination,
        )

    monkeypatch.setattr(torch.onnx, "export", wrong_interface)

    with pytest.raises(ValueError, match="input name"):
        exporter.export_reference(tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path)

    assert final.read_bytes() == b"known-good"
    assert not list(tmp_path.glob(".efficientphys-export-*"))


def test_export_rejects_custom_domain_nodes_and_opset_imports(tmp_path, monkeypatch):
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _FakeModel())

    def custom_domain(_model, _frames, destination, **_kwargs):
        graph = helper.make_graph(
            [helper.make_node("PrivateIdentity", ["frames"], ["pulse"], domain="private")],
            "custom-domain",
            [helper.make_tensor_value_info("frames", TensorProto.FLOAT, INPUT_SHAPE)],
            [helper.make_tensor_value_info("pulse", TensorProto.FLOAT, OUTPUT_SHAPE)],
        )
        onnx.save(
            helper.make_model(
                graph,
                opset_imports=[
                    helper.make_opsetid("", 17),
                    helper.make_opsetid("private", 1),
                ],
            ),
            destination,
        )

    monkeypatch.setattr(torch.onnx, "export", custom_domain)

    with pytest.raises(ValueError, match="custom domain"):
        exporter.export_reference(tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path)

    assert not list(tmp_path.glob(".efficientphys-export-*"))


@pytest.mark.parametrize("existing_finals", [False, True])
def test_second_publication_replace_failure_rolls_back_both_finals(
    tmp_path, monkeypatch, existing_finals
):
    _install_fake_export(monkeypatch)
    final_onnx = tmp_path / "efficientphys_pure.onnx"
    final_output = tmp_path / "pytorch_pulse_float32.npy"
    if existing_finals:
        final_onnx.write_bytes(b"old-onnx")
        final_output.write_bytes(b"old-output")

    real_replace = exporter.os.replace
    publication_replaces = 0
    failure_injected = False

    def fail_second_publication(source, destination):
        nonlocal publication_replaces, failure_injected
        destination = Path(destination)
        source = Path(source)
        if (
            destination in (final_onnx, final_output)
            and source.parent.name.startswith(".efficientphys-export-")
            and not source.name.startswith("backup-")
        ):
            publication_replaces += 1
            if publication_replaces == 2 and not failure_injected:
                failure_injected = True
                raise OSError("injected second publication failure")
        return real_replace(source, destination)

    monkeypatch.setattr(exporter.os, "replace", fail_second_publication)

    with pytest.raises(OSError, match="second publication failure"):
        exporter.export_reference(
            tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path
        )

    if existing_finals:
        assert final_onnx.read_bytes() == b"old-onnx"
        assert final_output.read_bytes() == b"old-output"
    else:
        assert not final_onnx.exists()
        assert not final_output.exists()
    assert not list(tmp_path.glob(".efficientphys-export-*"))


def test_concurrent_exports_use_unique_same_directory_staging_and_clean_it(
    tmp_path, monkeypatch
):
    destinations = []
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _FakeModel())

    def fake_export(_model, _frames, destination, **_kwargs):
        destinations.append(Path(destination))
        _write_valid_onnx(Path(destination))

    monkeypatch.setattr(torch.onnx, "export", fake_export)

    def run_export():
        return exporter.export_reference(
            tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _index: run_export(), range(2)))

    assert len(set(destinations)) == 2
    assert all(path.parent.parent == tmp_path for path in destinations)
    assert all(path.parent.name.startswith(".efficientphys-export-")
               for path in destinations)
    assert all(result.onnx.is_file() and result.pytorch_output.is_file()
               for result in results)
    assert not list(tmp_path.glob(".efficientphys-export-*"))


def test_export_rejects_onnx_above_constant_tensor_byte_limit(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(exporter, "load_official_model", lambda *_: _FakeModel())
    monkeypatch.setattr(exporter, "MAX_CONSTANT_TENSOR_BYTES", 719)

    def oversized_constant(_model, _frames, destination, **_kwargs):
        tensor = helper.make_tensor(
            "oversized",
            TensorProto.FLOAT,
            OUTPUT_SHAPE,
            np.zeros(OUTPUT_SHAPE, dtype=np.float32).tobytes(),
            raw=True,
        )
        graph = helper.make_graph(
            [helper.make_node("Constant", [], ["pulse"], value=tensor)],
            "oversized-constant",
            [helper.make_tensor_value_info("frames", TensorProto.FLOAT, INPUT_SHAPE)],
            [helper.make_tensor_value_info("pulse", TensorProto.FLOAT, OUTPUT_SHAPE)],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]),
            destination,
        )

    monkeypatch.setattr(torch.onnx, "export", oversized_constant)

    with pytest.raises(ValueError, match="constant tensor bytes.*720.*719"):
        exporter.export_reference(
            tmp_path, tmp_path / "checkpoint", _valid_frames(), tmp_path
        )

    assert not (tmp_path / "efficientphys_pure.onnx").exists()
    assert not (tmp_path / "pytorch_pulse_float32.npy").exists()
    assert not list(tmp_path.glob(".efficientphys-export-*"))


def test_cli_failure_writes_atomic_failure_report_from_outside_repo(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools" / "model_export" / "export_efficientphys.py"
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    np.save(
        artifact_dir / "frames_float32.npy",
        np.zeros((1,), dtype=np.float32),
        allow_pickle=False,
    )

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
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    report = json.loads((artifact_dir / "export_report.json").read_text())
    assert report["schema_version"] == 1
    assert report["passed"] is False
    assert "shape" in report["error"]
    assert report["exporter_mode"] == (
        "legacy_dynamo_false_with_explicit_aten_diff_slice_sub_lowering"
    )
    assert not (artifact_dir / "export_report.json.tmp").exists()


def test_main_success_report_binds_the_exact_published_onnx(
    tmp_path, monkeypatch
):
    _install_fake_export(monkeypatch)
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    np.save(
        artifact_dir / "frames_float32.npy",
        _valid_frames(),
        allow_pickle=False,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_efficientphys.py",
            "--toolbox",
            str(tmp_path / "toolbox"),
            "--checkpoint",
            str(tmp_path / "checkpoint"),
            "--artifact-dir",
            str(artifact_dir),
        ],
    )

    assert exporter.main() == 0

    onnx_path = artifact_dir / "efficientphys_pure.onnx"
    report = json.loads((artifact_dir / "export_report.json").read_text())
    assert report == {
        "schema_version": 1,
        "passed": True,
        "exporter_mode": exporter.EXPORTER_MODE,
        "qnn_conversion": "not_run",
        "qnn_risk": exporter.QNN_RISK,
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_metrics": {
            "constant_tensor_bytes": 0,
            "constant_tensor_byte_limit": exporter.MAX_CONSTANT_TENSOR_BYTES,
            "largest_constant_tensors": [],
            "scatter_nd_nodes": 0,
        },
    }


def _integration_resource(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        pytest.skip(f"{variable} is required for the export integration test")
    return Path(value)


@pytest.mark.integration
def test_cli_exports_fixed_official_model_from_outside_repo(tmp_path):
    toolbox = _integration_resource("RPPG_TOOLBOX_PATH")
    checkpoint = _integration_resource("RPPG_EFFICIENTPHYS_CHECKPOINT")
    source_artifact_dir = _integration_resource("RPPG_MODEL_ARTIFACT_DIR")
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools" / "model_export" / "export_efficientphys.py"
    artifact_dir = tmp_path / "fixed-artifact"
    artifact_dir.mkdir()
    frames = np.load(
        source_artifact_dir / "frames_float32.npy",
        allow_pickle=False,
    )
    np.save(artifact_dir / "frames_float32.npy", frames, allow_pickle=False)

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
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    onnx_path = artifact_dir / "efficientphys_pure.onnx"
    output_path = artifact_dir / "pytorch_pulse_float32.npy"
    assert str(onnx_path) in result.stdout
    assert str(output_path) in result.stdout
    report = json.loads((artifact_dir / "export_report.json").read_text())
    assert report == {
        "schema_version": 1,
        "passed": True,
        "exporter_mode": (
            "legacy_dynamo_false_with_explicit_aten_diff_slice_sub_lowering"
        ),
        "qnn_conversion": "not_run",
        "qnn_risk": (
            "requires_qairt_validation_due_to_12_scatternd_nodes_and_"
            "approximately_215mb_constants"
        ),
        "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "onnx_metrics": {
            "constant_tensor_bytes": 215315552,
            "constant_tensor_byte_limit": exporter.MAX_CONSTANT_TENSOR_BYTES,
            "largest_constant_tensors": [
                {
                    "bytes": 119439360,
                    "data_type": "FLOAT",
                    "node_name": "/TSM_2/Constant_1",
                    "shape": [18, 10, 32, 72, 72],
                    "tensor_name": "",
                },
                {
                    "bytes": 56448000,
                    "data_type": "FLOAT",
                    "node_name": "/TSM_4/Constant_1",
                    "shape": [18, 10, 64, 35, 35],
                    "tensor_name": "",
                },
                {
                    "bytes": 28224000,
                    "data_type": "FLOAT",
                    "node_name": "/TSM_3/Constant_1",
                    "shape": [18, 10, 32, 35, 35],
                    "tensor_name": "",
                },
                {
                    "bytes": 11197440,
                    "data_type": "FLOAT",
                    "node_name": "/TSM_1/Constant_1",
                    "shape": [18, 10, 3, 72, 72],
                    "tensor_name": "",
                },
            ],
            "scatter_nd_nodes": 12,
        },
    }

    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    assert [(value.name, [dim.dim_value for dim in value.type.tensor_type.shape.dim])
            for value in model.graph.input] == [("frames", list(INPUT_SHAPE))]
    assert [(value.name, [dim.dim_value for dim in value.type.tensor_type.shape.dim])
            for value in model.graph.output] == [("pulse", list(OUTPUT_SHAPE))]
    assert model.graph.input[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert [(item.domain, item.version) for item in model.opset_import] == [("", 17)]
    assert all(node.domain == "" for node in model.graph.node)
    assert "Slice" in [node.op_type for node in model.graph.node]
    assert "Sub" in [node.op_type for node in model.graph.node]
    output = np.load(output_path, allow_pickle=False)
    assert output.shape == OUTPUT_SHAPE
    assert output.dtype == np.float32
    assert np.isfinite(output).all()
    assert hashlib.sha256(output.tobytes()).hexdigest() == (
        "5c7d6202e56f02d4727571afbae3048368566e6d8b549c5d55544408282f5a56"
    )
