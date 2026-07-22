import hashlib
import importlib
from pathlib import Path
import subprocess
import sys

import numpy as np

from tools.model_export.efficientphys_reference import prepare_model_input


EXPECTED_SOURCE_SHA256 = (
    "dca274b1e509762f17a2dff152fa0fc3fa87c0c6f2726db284792d3494350b46"
)
EXPECTED_FRAMES_SHA256 = (
    "b685ed04d5b6a96bbdd2f97fed6cc04a8a2eb35c38b7ebc820eb1a37af2d2f40"
)


def raw_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def test_synthetic_source_and_tensor_are_bit_reproducible():
    generator = importlib.import_module("tools.model_export.generate_test_vector")

    first_source = generator.build_source_window()
    second_source = generator.build_source_window()

    assert generator.GENERATOR_SEED == 20260722
    assert first_source.tobytes(order="C") == second_source.tobytes(order="C")
    assert first_source.shape == (180, 72, 72, 3)
    assert first_source.dtype == np.uint8
    assert first_source.flags.c_contiguous
    assert raw_sha256(first_source) == EXPECTED_SOURCE_SHA256

    frames = prepare_model_input(first_source)
    assert frames.shape == (181, 3, 72, 72)
    assert frames.dtype == np.float32
    assert frames.flags.c_contiguous
    assert raw_sha256(frames) == EXPECTED_FRAMES_SHA256


def test_cli_runs_outside_repo_and_writes_only_below_artifact_dir(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    script = repo_root / "tools" / "model_export" / "generate_test_vector.py"
    outside_cwd = tmp_path / "outside-cwd"
    outside_cwd.mkdir()
    artifact_dir = tmp_path / "nested" / "artifacts"

    outside_source = tmp_path / "source_rgb_uint8.npy"
    outside_frames = tmp_path / "frames_float32.npy"
    outside_source.write_bytes(b"keep source")
    outside_frames.write_bytes(b"keep frames")

    subprocess.run(
        [sys.executable, str(script), "--artifact-dir", str(artifact_dir)],
        cwd=outside_cwd,
        check=True,
    )

    source = np.load(artifact_dir / "source_rgb_uint8.npy", allow_pickle=False)
    frames = np.load(artifact_dir / "frames_float32.npy", allow_pickle=False)
    assert source.shape == (180, 72, 72, 3)
    assert source.dtype == np.uint8
    assert source.flags.c_contiguous
    assert raw_sha256(source) == EXPECTED_SOURCE_SHA256
    assert frames.shape == (181, 3, 72, 72)
    assert frames.dtype == np.float32
    assert frames.flags.c_contiguous
    assert raw_sha256(frames) == EXPECTED_FRAMES_SHA256
    assert outside_source.read_bytes() == b"keep source"
    assert outside_frames.read_bytes() == b"keep frames"
