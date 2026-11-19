"""Tests that `spectrogram()` gives 0.1.3's pixels."""

import numpy as np
import pytest
import tensorflow as tf

from sox_tensorflow import spectrogram
from tests.reference_cases import (
    get_platform_key,
    get_reference_cases,
    get_reference_path,
)


RELEASE_PLATFORMS = ("darwin-arm64", "linux-x86_64")
CASES = get_reference_cases()


@pytest.fixture(scope="session")
def reference():
    """The 0.1.3 pixels for this platform, by case name."""
    path = get_reference_path()
    if not path.exists():
        message = f"no reference for this platform: {path}"
        if get_platform_key() in RELEASE_PLATFORMS:
            pytest.fail(message)
        pytest.skip(message)
    with np.load(path) as data:
        return {name: data[name] for name in data.files}


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_spectrogram_matches_the_0_1_3_reference(case, reference):
    pixels = spectrogram(case.get_input(), **case.kwargs)
    assert pixels.dtype == tf.uint8
    np.testing.assert_array_equal(pixels.numpy(), reference[case.name])
