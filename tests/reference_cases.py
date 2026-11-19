"""
The inputs that the 0.1.3 reference covers.

Each case makes a new input every time it is asked, so that no test can
change the input of another.
"""

from __future__ import annotations

import pathlib
import platform
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import tensorflow as tf


SHAPE = (257, 1000)
FIXTURES = pathlib.Path(__file__).parent / "fixtures"


@dataclass(frozen=True, eq=False)
class ReferenceCase:
    """One input and the keyword arguments to pass to `spectrogram()`."""
    name: str
    get_input: Callable[[], object]
    kwargs: dict = field(default_factory=dict)


def make_noise(n: int) -> np.ndarray:
    """Uniform noise in [-0.5, 0.5), float64, the same for every call."""
    return np.random.default_rng(7).uniform(-0.5, 0.5, n)


def make_square(n: int, rate: int) -> np.ndarray:
    """A 1 kHz square wave at half scale, float64."""
    t = np.arange(n) / rate
    return 0.5 * np.sign(np.sin(2 * np.pi * 1000 * t))


def make_case(name: str, get_input: Callable[[], object], rate: int,
              **kwargs) -> ReferenceCase:
    """A case at the default shape and `rate`, unless `kwargs` says otherwise."""
    return ReferenceCase(
        name=name,
        get_input=get_input,
        kwargs={"shape": SHAPE, "sample_rate": rate, **kwargs},
    )


def get_reference_cases() -> list[ReferenceCase]:
    """All 28 reference cases, in a fixed order."""
    cases = []

    # Noise and a square wave, 12 s, at five input rates.
    for signal in ("noise", "square"):
        for rate in (8000, 22050, 44100, 48000, 96000):
            n = 12 * rate
            if signal == "noise":
                get_input = (lambda n=n: make_noise(n))
            else:
                get_input = (lambda n=n, rate=rate: make_square(n, rate))
            cases.append(make_case(f"{signal}-{rate}", get_input, rate))

    # Silence gives index 4 everywhere at every rate, so one rate is enough.
    cases.append(make_case("silence-48000", lambda: np.zeros(576000), 48000))

    # Other lengths.
    cases.append(make_case("noise-48000-3s", lambda: make_noise(3 * 48000), 48000))
    cases.append(make_case("noise-48000-20s", lambda: make_noise(20 * 48000), 48000))
    cases.append(make_case("noise-48000-11.99s",
                           lambda: make_noise(round(11.99 * 48000)), 48000))

    # Another shape, and another output rate.
    cases.append(make_case("noise-48000-shape-129x500",
                           lambda: make_noise(576000), 48000, shape=(129, 500)))
    cases.append(make_case("noise-48000-out-16000",
                           lambda: make_noise(576000), 48000,
                           output_sample_rate=16000))

    # Rounding edges of the step size. At 8 kHz there is no resample, so the
    # length is exact, and block_steps_float is length / 1000.
    #   96500: 96.5, an exact half, rounds to even (step 96).
    #   97500: 97.49999999999999 in float64, so step 97, not 98.
    #   256000 and 256001: the ceil goes from 1 to 2, the step from 256 to 128.
    for n in (96500, 97500, 256000, 256001):
        cases.append(make_case(f"noise-8000-n{n}", lambda n=n: make_noise(n), 8000))

    # Another dB range.
    cases.append(make_case("noise-48000-db-range-60",
                           lambda: make_noise(576000), 48000, db_range=60))

    # Input kinds. 0.1.3 casts int32 to float without scaling; the case keeps that.
    cases.append(make_case("noise-48000-float32",
                           lambda: make_noise(576000).astype(np.float32), 48000))
    cases.append(make_case("noise-48000-int32",
                           lambda: (make_noise(576000) * 2**31).astype(np.int32),
                           48000))
    cases.append(make_case("noise-48000-tensor-float32",
                           lambda: tf.constant(make_noise(576000), dtype=tf.float32),
                           48000))
    cases.append(make_case("noise-48000-tensor-float64",
                           lambda: tf.constant(make_noise(576000), dtype=tf.float64),
                           48000))

    # A segment of a longer input.
    cases.append(make_case("noise-48000-segment",
                           lambda: make_noise(30 * 48000), 48000,
                           segment=1, segment_duration=12, segment_overlap=2))

    # Loud noise clips after the resample. A constant 2.0 clips to 1.0 and
    # reaches index 254 (the dBfs >= 0 branch).
    cases.append(make_case("loud-48000", lambda: make_noise(576000) * 4, 48000))
    cases.append(make_case("saturated-48000", lambda: np.full(576000, 2.0), 48000))

    return cases


def get_platform_key() -> str:
    """The name of this platform, such as `darwin-arm64` or `linux-x86_64`."""
    return f"{sys.platform}-{platform.machine()}"


def get_reference_path() -> pathlib.Path:
    """The reference file for this platform."""
    return FIXTURES / f"reference-{get_platform_key()}.npz"
