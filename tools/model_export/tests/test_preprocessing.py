import numpy as np
import pytest

from tools.model_export.efficientphys_reference import (
    EXPECTED_CHECKPOINT_SHA256 as ACTUAL_CHECKPOINT_SHA256,
    FRAME_DEPTH,
    FRAME_RATE,
    MODEL_INPUT_SHAPE,
    MODEL_OUTPUT_SHAPE,
    SOURCE_SHAPE,
    prepare_model_input,
    sha256_file,
)


EXPECTED_SOURCE_SHAPE = (180, 72, 72, 3)
EXPECTED_MODEL_INPUT_SHAPE = (181, 3, 72, 72)
EXPECTED_MODEL_OUTPUT_SHAPE = (180, 1)
EXPECTED_CHECKPOINT_SHA256 = (
    "e65a962e07bcac32a668e6acb9f8ed43cdb1b01cfb97262654dc5b55c0cf3a49"
)


def _toolbox_expected_first_180(source_rgb):
    source_float64 = source_rgb.astype(np.float64)
    standard_deviation = source_float64.std(ddof=0)
    if standard_deviation == 0.0 or not np.isfinite(standard_deviation):
        standardized = np.zeros_like(source_float64)
    else:
        standardized = (
            source_float64 - source_float64.mean()
        ) / standard_deviation
        standardized[~np.isfinite(standardized)] = 0.0
    return np.transpose(standardized.astype(np.float32), (0, 3, 1, 2))


def test_export_contract_constants():
    assert SOURCE_SHAPE == EXPECTED_SOURCE_SHAPE
    assert MODEL_INPUT_SHAPE == EXPECTED_MODEL_INPUT_SHAPE
    assert MODEL_OUTPUT_SHAPE == EXPECTED_MODEL_OUTPUT_SHAPE
    assert FRAME_RATE == 30.0
    assert FRAME_DEPTH == 10
    assert ACTUAL_CHECKPOINT_SHA256 == EXPECTED_CHECKPOINT_SHA256


def test_prepare_model_input_standardizes_globally_transposes_and_duplicates_last_frame():
    values = np.arange(np.prod(EXPECTED_SOURCE_SHAPE), dtype=np.uint32)
    frames = (values % 256).astype(np.uint8).reshape(EXPECTED_SOURCE_SHAPE)

    expected_first_180 = _toolbox_expected_first_180(frames)

    actual = prepare_model_input(frames)

    assert actual.shape == EXPECTED_MODEL_INPUT_SHAPE
    assert actual.dtype == np.float32
    assert actual.flags.c_contiguous
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(actual[:180], expected_first_180, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(actual[180], actual[179])
    assert float(actual[:180].mean()) == pytest.approx(0.0, abs=1e-6)
    assert float(actual[:180].std(ddof=0)) == pytest.approx(1.0, abs=1e-6)


def test_prepare_model_input_matches_float64_toolbox_oracle_for_pathological_distribution():
    frames = np.zeros(EXPECTED_SOURCE_SHAPE, dtype=np.uint8)
    frames.reshape(-1)[:3] = 255
    expected_first_180 = _toolbox_expected_first_180(frames)

    actual = prepare_model_input(frames)

    maximum_absolute_error = float(
        np.max(np.abs(actual[:180] - expected_first_180))
    )
    assert maximum_absolute_error <= 1e-6


def test_prepare_model_input_returns_finite_zeros_for_zero_variance_input():
    frames = np.full(EXPECTED_SOURCE_SHAPE, 37, dtype=np.uint8)

    actual = prepare_model_input(frames)

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
def test_prepare_model_input_rejects_incorrect_shape(actual_shape):
    frames = np.zeros(actual_shape, dtype=np.uint8)

    with pytest.raises(ValueError) as error:
        prepare_model_input(frames)

    message = str(error.value)
    assert "shape" in message.lower()
    assert str(actual_shape) in message
    assert str(EXPECTED_SOURCE_SHAPE) in message


def test_prepare_model_input_rejects_non_uint8_dtype():
    frames = np.zeros(EXPECTED_SOURCE_SHAPE, dtype=np.float32)

    with pytest.raises(TypeError) as error:
        prepare_model_input(frames)

    message = str(error.value)
    assert "dtype" in message.lower()
    assert "float32" in message
    assert "uint8" in message


def test_sha256_file_returns_lowercase_hex_digest_for_small_file(tmp_path):
    source = tmp_path / "small.bin"
    source.write_bytes(b"abc")

    digest = sha256_file(source)

    assert digest == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert digest == digest.lower()
    assert len(digest) == 64
