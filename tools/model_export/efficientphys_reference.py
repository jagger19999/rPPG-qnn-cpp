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


def preprocess_frames(frames: np.ndarray) -> np.ndarray:
    """Standardize an RGB frame window and append its final frame."""
    if not isinstance(frames, np.ndarray):
        raise TypeError(
            "frames must be a NumPy array with dtype uint8; "
            f"got {type(frames).__name__}"
        )
    if frames.dtype != np.uint8:
        raise TypeError(f"frames dtype must be uint8; got {frames.dtype}")
    if frames.shape != SOURCE_SHAPE:
        raise ValueError(
            f"frames shape must be {SOURCE_SHAPE}; got {frames.shape}"
        )

    float_frames = frames.astype(np.float32)
    standard_deviation = float_frames.std(ddof=0)
    if standard_deviation == 0.0:
        standardized = np.zeros_like(float_frames)
    else:
        standardized = (
            float_frames - float_frames.mean()
        ) / standard_deviation

    output = np.empty(MODEL_INPUT_SHAPE, dtype=np.float32)
    output[: SOURCE_SHAPE[0]] = np.transpose(standardized, (0, 3, 1, 2))
    output[SOURCE_SHAPE[0]] = output[SOURCE_SHAPE[0] - 1]
    return output


def sha256_file(path: str | PathLike[str]) -> str:
    """Return the lowercase SHA-256 hex digest of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
