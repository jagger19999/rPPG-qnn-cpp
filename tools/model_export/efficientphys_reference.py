"""Reference constants and preprocessing for EfficientPhys model export."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import importlib.util
from os import PathLike
from pathlib import Path

import numpy as np
import torch


SOURCE_SHAPE = (180, 72, 72, 3)
MODEL_INPUT_SHAPE = (181, 3, 72, 72)
MODEL_OUTPUT_SHAPE = (180, 1)
FRAME_RATE = 30.0
FRAME_DEPTH = 10
EXPECTED_CHECKPOINT_SHA256 = (
    "e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49"
)


def normalize_module_prefix(state: Mapping) -> OrderedDict:
    """Remove exactly one required ``module.`` prefix from every state key."""
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint state must be a Mapping")
    if not state:
        raise ValueError("checkpoint state must be non-empty")

    keys = list(state.keys())
    if any(not isinstance(key, str) for key in keys):
        raise TypeError("every checkpoint state key must be a string")
    if not all(key.startswith("module.") for key in keys):
        raise ValueError("every key must have exactly one 'module.' prefix")
    if any(key.startswith("module.module.") for key in keys):
        raise ValueError("every key must have exactly one 'module.' prefix")

    normalized = OrderedDict()
    prefix_length = len("module.")
    for key, value in state.items():
        normalized_key = key[prefix_length:]
        if not normalized_key:
            raise ValueError("removing 'module.' must not produce an empty key")
        if normalized_key in normalized:
            raise ValueError(f"normalizing checkpoint keys produced a collision: {key}")
        normalized[normalized_key] = value
    return normalized


def _load_model_module(toolbox: Path):
    """Load EfficientPhys only from the requested Toolbox source tree."""
    source = Path(toolbox) / "neural_methods" / "model" / "EfficientPhys.py"
    if not source.is_file():
        raise FileNotFoundError(f"official EfficientPhys source file not found: {source}")

    spec = importlib.util.spec_from_file_location(
        "_rppg_toolbox_official_efficientphys", source
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_official_model(toolbox: Path, checkpoint: Path):
    """Strictly load the pinned official EfficientPhys checkpoint on CPU."""
    checkpoint = Path(checkpoint)
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError(
            "checkpoint SHA-256 mismatch: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {actual_sha256}"
        )

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError("checkpoint must contain a Mapping state dict")
    normalized_state = normalize_module_prefix(state)

    model_module = _load_model_module(Path(toolbox))
    model = model_module.EfficientPhys(frame_depth=FRAME_DEPTH, img_size=72)
    model.load_state_dict(normalized_state, strict=True)
    model.cpu()
    model.eval()
    return model


def prepare_model_input(source_rgb: np.ndarray) -> np.ndarray:
    """Reproduce Toolbox preprocessing for one RGB source window."""
    if not isinstance(source_rgb, np.ndarray):
        raise TypeError(
            "source_rgb must be a NumPy array with dtype uint8; "
            f"got {type(source_rgb).__name__}"
        )
    if source_rgb.dtype != np.uint8:
        raise TypeError(
            f"source_rgb dtype must be uint8; got {source_rgb.dtype}"
        )
    if source_rgb.shape != SOURCE_SHAPE:
        raise ValueError(
            f"source_rgb shape must be {SOURCE_SHAPE}; got {source_rgb.shape}"
        )

    source_float64 = source_rgb.astype(np.float64)
    mean = source_float64.mean()
    standard_deviation = source_float64.std(ddof=0)
    if (
        standard_deviation == 0.0
        or not np.isfinite(mean)
        or not np.isfinite(standard_deviation)
    ):
        standardized = np.zeros_like(source_float64)
    else:
        standardized = (source_float64 - mean) / standard_deviation
        standardized[~np.isfinite(standardized)] = 0.0

    source_tchw = np.transpose(
        standardized.astype(np.float32), (0, 3, 1, 2)
    )
    model_input = np.concatenate((source_tchw, source_tchw[-1:]), axis=0)
    model_input = np.ascontiguousarray(model_input)

    if model_input.shape != MODEL_INPUT_SHAPE:
        raise RuntimeError(
            "prepared model input shape must be "
            f"{MODEL_INPUT_SHAPE}; got {model_input.shape}"
        )
    if not np.isfinite(model_input).all():
        raise RuntimeError("prepared model input must contain only finite values")
    return model_input


def sha256_file(path: str | PathLike[str]) -> str:
    """Return the lowercase SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
