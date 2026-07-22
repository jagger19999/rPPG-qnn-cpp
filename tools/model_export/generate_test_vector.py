"""Generate the deterministic EfficientPhys source and model-input arrays."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.model_export.efficientphys_reference import prepare_model_input


GENERATOR_SEED = 20260722


def build_source_window() -> np.ndarray:
    """Return the frozen synthetic RGB source window."""
    t = np.arange(180, dtype=np.uint32)[:, None, None, None]
    y = np.arange(72, dtype=np.uint32)[None, :, None, None]
    x = np.arange(72, dtype=np.uint32)[None, None, :, None]
    c = np.arange(3, dtype=np.uint32)[None, None, None, :]
    return (
        (
            GENERATOR_SEED
            + t * 17
            + y * 13
            + x * 7
            + c * 53
            + (t * y) % 251
        )
        % 256
    ).astype(np.uint8)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    source = build_source_window()
    frames = prepare_model_input(source)
    np.save(
        args.artifact_dir / "source_rgb_uint8.npy",
        source,
        allow_pickle=False,
    )
    np.save(
        args.artifact_dir / "frames_float32.npy",
        frames,
        allow_pickle=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
