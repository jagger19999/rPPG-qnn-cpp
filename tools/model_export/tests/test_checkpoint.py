from collections import OrderedDict
from collections.abc import Mapping
import os
from pathlib import Path

import pytest
import torch

from tools.model_export import efficientphys_reference as reference


def test_normalize_module_prefix_removes_exactly_one_prefix_and_preserves_order():
    first = object()
    second = object()
    state = OrderedDict(
        (("module.first", first), ("module.layer.second", second))
    )

    normalized = reference.normalize_module_prefix(state)

    assert isinstance(normalized, OrderedDict)
    assert list(normalized) == ["first", "layer.second"]
    assert normalized["first"] is first
    assert normalized["layer.second"] is second


def test_normalize_module_prefix_rejects_mixed_prefixed_and_unprefixed_keys():
    state = OrderedDict(
        (("module.first", torch.tensor(1.0)), ("second", torch.tensor(2.0)))
    )

    with pytest.raises(ValueError, match="every key"):
        reference.normalize_module_prefix(state)


def test_normalize_module_prefix_rejects_multiple_prefix_layers():
    state = OrderedDict((("module.module.first", torch.tensor(1.0)),))

    with pytest.raises(ValueError, match="exactly one"):
        reference.normalize_module_prefix(state)


@pytest.mark.parametrize(
    ("state", "error_type", "message"),
    [
        ({}, ValueError, "non-empty"),
        (["module.first"], TypeError, "Mapping"),
        ({1: torch.tensor(1.0)}, TypeError, "string"),
        ({"module.": torch.tensor(1.0)}, ValueError, "empty"),
    ],
)
def test_normalize_module_prefix_rejects_invalid_state(state, error_type, message):
    with pytest.raises(error_type, match=message):
        reference.normalize_module_prefix(state)


def test_load_model_module_reports_missing_exact_official_source(tmp_path):
    expected_source = tmp_path / "neural_methods" / "model" / "EfficientPhys.py"

    with pytest.raises(FileNotFoundError) as error:
        reference._load_model_module(tmp_path)

    assert str(expected_source) in str(error.value)


def _write_fake_official_model(toolbox: Path) -> None:
    source = toolbox / "neural_methods" / "model" / "EfficientPhys.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        """
import torch


class EfficientPhys(torch.nn.Module):
    def __init__(self, frame_depth, img_size):
        super().__init__()
        self.frame_depth = frame_depth
        self.img_size = img_size
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def forward(self, inputs):
        return inputs
""".lstrip(),
        encoding="utf-8",
    )


def test_load_official_model_rejects_hash_mismatch_before_torch_load(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"not the official checkpoint")
    load_called = False

    def unexpected_load(*args, **kwargs):
        nonlocal load_called
        load_called = True
        raise AssertionError("torch.load must not run after a hash mismatch")

    monkeypatch.setattr(torch, "load", unexpected_load)

    with pytest.raises(ValueError, match="SHA-256"):
        reference.load_official_model(tmp_path, checkpoint)

    assert not load_called


def test_load_official_model_requires_checkpoint_mapping(tmp_path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr(
        reference, "sha256_file", lambda path: reference.EXPECTED_CHECKPOINT_SHA256
    )
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: ["not", "a", "mapping"])

    with pytest.raises(TypeError, match="Mapping"):
        reference.load_official_model(tmp_path, checkpoint)


def test_load_official_model_loads_strictly_on_cpu_in_eval_mode(tmp_path, monkeypatch):
    _write_fake_official_model(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr(
        reference, "sha256_file", lambda path: reference.EXPECTED_CHECKPOINT_SHA256
    )
    load_arguments = {}

    def fake_load(path, **kwargs):
        load_arguments["path"] = path
        load_arguments.update(kwargs)
        return OrderedDict((("module.weight", torch.tensor([7.0])),))

    monkeypatch.setattr(torch, "load", fake_load)

    model = reference.load_official_model(tmp_path, checkpoint)

    assert load_arguments == {
        "path": checkpoint,
        "map_location": "cpu",
        "weights_only": True,
    }
    assert model.frame_depth == 10
    assert model.img_size == 72
    assert not model.training
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    torch.testing.assert_close(model.weight, torch.tensor([7.0]))


def test_load_official_model_propagates_strict_state_dict_errors(tmp_path, monkeypatch):
    _write_fake_official_model(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"stub")
    monkeypatch.setattr(
        reference, "sha256_file", lambda path: reference.EXPECTED_CHECKPOINT_SHA256
    )
    monkeypatch.setattr(
        torch,
        "load",
        lambda *args, **kwargs: OrderedDict(
            (("module.unexpected", torch.tensor([7.0])),)
        ),
    )

    with pytest.raises(RuntimeError) as error:
        reference.load_official_model(tmp_path, checkpoint)

    assert "Missing key" in str(error.value)
    assert "Unexpected key" in str(error.value)


def _integration_resource(variable: str) -> Path:
    value = os.environ.get(variable)
    if value is None:
        pytest.skip(f"{variable} is required for the checkpoint integration test")
    return Path(value)


@pytest.mark.integration
def test_official_checkpoint_loads_and_runs_inference():
    toolbox = _integration_resource("RPPG_TOOLBOX_PATH")
    checkpoint = _integration_resource("RPPG_EFFICIENTPHYS_CHECKPOINT")

    assert reference.sha256_file(checkpoint) == reference.EXPECTED_CHECKPOINT_SHA256
    model = reference.load_official_model(toolbox, checkpoint)

    assert isinstance(model.state_dict(), Mapping)
    assert not model.training
    assert all(parameter.device.type == "cpu" for parameter in model.parameters())
    with torch.inference_mode():
        output = model(torch.zeros((181, 3, 72, 72), dtype=torch.float32))

    assert output.shape == (180, 1)
    assert torch.isfinite(output).all()
