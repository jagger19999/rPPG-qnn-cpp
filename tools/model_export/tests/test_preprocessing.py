import importlib

import numpy as np
import pytest


EXPECTED_SOURCE_SHAPE = (180, 72, 72, 3)
EXPECTED_MODEL_INPUT_SHAPE = (181, 3, 72, 72)
EXPECTED_MODEL_OUTPUT_SHAPE = (180, 1)
EXPECTED_CHECKPOINT_SHA256 = (
    "e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49"
)


def _reference_module():
    return importlib.import_module("tools.model_export.efficientphys_reference")


def test_export_contract_constants():
    reference = _reference_module()

    assert reference.SOURCE_SHAPE == EXPECTED_SOURCE_SHAPE
    assert reference.MODEL_INPUT_SHAPE == EXPECTED_MODEL_INPUT_SHAPE
    assert reference.MODEL_OUTPUT_SHAPE == EXPECTED_MODEL_OUTPUT_SHAPE
    assert reference.FRAME_RATE == 30.0
    assert reference.FRAME_DEPTH == 10
    assert reference.EXPECTED_CHECKPOINT_SHA256 == EXPECTED_CHECKPOINT_SHA256


def test_preprocess_frames_standardizes_globally_transposes_and_duplicates_last_frame():
    reference = _reference_module()
    values = np.arange(np.prod(EXPECTED_SOURCE_SHAPE), dtype=np.uint32)
    frames = (values % 256).astype(np.uint8).reshape(EXPECTED_SOURCE_SHAPE)

    float_frames = frames.astype(np.float32)
    expected_thwc = (float_frames - float_frames.mean()) / float_frames.std(ddof=0)
    expected_first_180 = np.transpose(expected_thwc, (0, 3, 1, 2))

    actual = reference.preprocess_frames(frames)

    assert actual.shape == EXPECTED_MODEL_INPUT_SHAPE
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[:180], expected_first_180, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(actual[180], actual[179])
    assert float(actual[:180].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(actual[:180].std(ddof=0)) == pytest.approx(1.0, abs=1e-6)


def test_preprocess_frames_returns_finite_zeros_for_zero_variance_input():
    reference = _reference_module()
    frames = np.full(EXPECTED_SOURCE_SHAPE, 37, dtype=np.uint8)

    actual = reference.preprocess_frames(frames)

    assert actual.shape == EXPECTED_MODEL_INPUT_SHAPE
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual, np.zeros(EXPECTED_MODEL_INPUT_SHAPE, dtype=np.float32))


@pytest.mark.parametrize(
    "actual_shape",
    [
        (179, 72, 72, 3),
        (181, 72, 72, 3),
        (180, 72, 72, 1),
    ],
)
def test_preprocess_frames_rejects_incorrect_shape(actual_shape):
    reference = _reference_module()
    frames = np.zeros(actual_shape, dtype=np.uint8)

    with pytest.raises(ValueError) as error:
        reference.preprocess_frames(frames)

    message = str(error.value)
    assert "shape" in message.lower()
    assert str(actual_shape) in message
    assert str(EXPECTED_SOURCE_SHAPE) in message


def test_preprocess_frames_rejects_non_uint8_dtype():
    reference = _reference_module()
    frames = np.zeros(EXPECTED_SOURCE_SHAPE, dtype=np.float32)

    with pytest.raises(TypeError) as error:
        reference.preprocess_frames(frames)

    message = str(error.value)
    assert "dtype" in message.lower()
    assert "float32" in message
    assert "uint8" in message


def test_sha256_file_returns_lowercase_hex_digest_for_small_file(tmp_path):
    reference = _reference_module()
    source = tmp_path / "small.bin"
    source.write_bytes(b"abc")

    digest = reference.sha256_file(source)

    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert digest == digest.lower()
    assert len(digest) == 64
