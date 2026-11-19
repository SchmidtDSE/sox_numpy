"""Tests that `spectrogram()` gives 0.1.3's pixels."""

import contextlib
import importlib.metadata
import random
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import tensorflow as tf
from PIL import Image

from sox_numpy import processor, spectrogram
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


CASES_BY_NAME = {case.name: case for case in CASES}


def get_pixels(name):
    """The pixels of one reference case, as a NumPy array."""
    case = CASES_BY_NAME[name]
    return spectrogram(case.get_input(), **case.kwargs).numpy()


def test_a_call_after_a_call_with_another_key_matches_a_fresh_call(reference):
    names = ["noise-48000", "noise-48000-3s", "noise-48000-shape-129x500",
             "noise-48000-out-16000"]
    for order in (names, names[::-1]):
        processor._get_setup.cache_clear()
        for name in order:
            np.testing.assert_array_equal(get_pixels(name), reference[name], err_msg=name)


def test_a_call_does_not_change_the_shared_setup(reference):
    setup = processor._get_setup(257, 1000, 96000, 8000)
    windows = setup.windows.copy()
    for name in ("noise-48000", "loud-48000"):
        np.testing.assert_array_equal(get_pixels(name), reference[name])
    assert setup.windows.flags.writeable is False
    np.testing.assert_array_equal(setup.windows, windows)
    assert processor._get_setup(257, 1000, 96000, 8000) is setup


def test_concurrent_calls_match_serial_calls(reference):
    names = [case.name for case in CASES] * 3
    random.Random(3).shuffle(names)
    processor._get_setup.cache_clear()
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(get_pixels, names))
    for name, pixels in zip(names, results):
        np.testing.assert_array_equal(pixels, reference[name], err_msg=name)


def test_the_setup_cache_holds_at_most_16_setups():
    # 4 of these 40 lengths (such as 81600) put the last frame past the padded
    # audio; 0.1.3 raises IndexError on them, 0.1.4 ValueError, and a call that
    # raises is not cached.
    processor._get_setup.cache_clear()
    made = 0
    for length in range(80000, 111200 + 1, 800):
        with contextlib.suppress(ValueError):
            processor._get_setup(257, 1000, length, 8000)
            made += 1
    assert made > 16
    assert processor._get_setup.cache_info().currsize <= 16


def test_spectrogram_does_not_change_the_input_array():
    case = CASES_BY_NAME["noise-48000"]
    audio = case.get_input()
    before = audio.copy()
    spectrogram(audio, **case.kwargs)
    np.testing.assert_array_equal(audio, before)


@pytest.mark.parametrize(
    "audio",
    [np.zeros(400), np.zeros(0), np.zeros(81600)],
    ids=["400", "empty", "81600-last-frame-past-the-end"],
)
def test_audio_too_short_for_the_width_raises_value_error(audio):
    with pytest.raises(ValueError):
        spectrogram(audio, shape=(257, 1000), sample_rate=8000)


def test_dest_writes_a_png_of_the_returned_pixels(tmp_path):
    case = CASES_BY_NAME["noise-48000"]
    path = tmp_path / "a.png"
    result = spectrogram(case.get_input(), dest=path, **case.kwargs)
    assert result == str(path)
    pixels = spectrogram(case.get_input(), **case.kwargs).numpy()
    np.testing.assert_array_equal(np.asarray(Image.open(path)), pixels)


def test_the_package_is_sox_numpy_0_1_0():
    assert importlib.metadata.version("sox_numpy") == "0.1.0"
