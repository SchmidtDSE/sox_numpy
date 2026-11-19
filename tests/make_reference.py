"""
Make the reference pixels for this platform.

Run it only on the code of tag 0.1.3, once for each platform:

    pixi run -e test python tests/make_reference.py

It writes tests/fixtures/reference-<platform>.npz, one array for each case in
`get_reference_cases()`, and `_versions` with the NumPy and TensorFlow versions.
"""

import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sox_numpy import spectrogram  # noqa: E402
from tests.reference_cases import get_reference_cases, get_reference_path  # noqa: E402


def make_reference() -> Path:
    """Compute every case and write the reference file."""
    arrays = {}
    for case in get_reference_cases():
        arrays[case.name] = spectrogram(case.get_input(), **case.kwargs).numpy()
        print(f"{case.name}: {arrays[case.name].shape} {arrays[case.name].dtype}")

    # The reference must reach the dBfs >= 0 branch.
    assert (arrays["saturated-48000"] == 254).any()

    arrays["_versions"] = np.array(f"numpy {np.__version__}, tensorflow {tf.__version__}")
    path = get_reference_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


if __name__ == "__main__":
    print(make_reference())
