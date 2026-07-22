"""Reference constants and preprocessing for EfficientPhys model export."""

from __future__ import annotations

import hashlib
from os import PathLike

import numpy as np


SOURCE_SHAPE = (180, 72, 72, 3)
MODEL_INPUT_SHAPE = (181, 3, 72, 72)
MODEL_OUTPUT_SHAPE = (180, 1)
FRAME_RATE = 30.0
FRAME_DEPTH = 10
EXPECTED_CHECKPOINT_SHA256 = (
    "e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49"
)


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
