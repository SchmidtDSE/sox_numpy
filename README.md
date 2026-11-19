# sox_numpy

SoX-style spectrograms in NumPy: the same images as the `sox` binary's `spectrogram`
effect, without calling `sox` and without TensorFlow. These are the spectrograms
that the [PNW-Cnet v4 model](https://github.com/zjruff/Shiny_PNW-Cnet) was trained on.

sox_numpy began as [sox_tensorflow](https://github.com/SchmidtDSE/sox_tensorflow) 0.1.3
and keeps its history. It keeps the same algorithm and parameters, with the FFT in NumPy.
It uses less CPU per spectrogram and agrees with `sox` at least as well as 0.1.3 (see
[Agreement with sox](#agreement-with-sox)).

**LINKS**

- sox: https://github.com/chirlu/sox
- audio samples:
    * https://dse-soundhub.s3.us-west-2.amazonaws.com/public/audio/dev/20230522_000000.flac
    * https://dse-soundhub.s3.us-west-2.amazonaws.com/public/audio/dev/20230526_000000.flac
- pnw-cnet-model: https://dse-soundhub.s3.us-west-2.amazonaws.com/public/models/pnw/PNW-Cnet_v4_TF.h5

---

## INSTALL

```bash
pip install sox_numpy          # NumPy, soundfile, soxr
pip install "sox_numpy[png]"   # also Pillow, to write PNGs with `dest`
```

Needs Python 3.11 or later and NumPy 2. NumPy 1 upcasts the FFT to complex128,
which is slower and gives different pixels.

## QUICK START

```python
import soundfile as sf
from sox_numpy import spectrogram, spectrogram_from_flac

# From a NumPy array (one channel)
samples, sr = sf.read('audio.flac', dtype='float64')
pixels = spectrogram(samples, shape=(257, 1000), sample_rate=sr)
pixels.shape, pixels.dtype   # (257, 1000), uint8

# From a FLAC file, straight to a PNG
path = spectrogram_from_flac(
    flac_path='audio.flac',
    shape=(257, 1000),
    duration=12.0,
    segment=0,
    dest='spectrogram.png'
)
```

---

## API

### `spectrogram(audio_array, shape, dest=None, ...)`

Makes a SoX-matching spectrogram from one channel of audio.

| Argument | Type | Description |
|---|---|---|
| `audio_array` | `np.ndarray`, or anything `np.asarray` takes | One channel of samples, float32/float64 in [-1, 1]. A `tf.Tensor` works without sox_numpy importing TensorFlow. A 2-D array raises `ValueError` |
| `shape` | `(int, int)` | Output shape as `(height, width)`. Height sets the frequency resolution: DFT size = 2 × (height − 1) |
| `dest` | `str` or `Path`, optional | Output PNG path (needs the `png` extra). If `None`, returns the array |
| `segment` | `int`, optional | Segment index (0-based) to take from the audio |
| `segment_duration` | `float`, optional | Duration of each segment in seconds |
| `segment_overlap` | `float`, optional | Overlap between segments in seconds |
| `sample_rate` | `int` | Sample rate of the input audio in Hz (required) |
| `output_sample_rate` | `int` | Sample rate for the spectrogram (default: 8000) |
| `create_parents` | `bool` | Make the parent directories of `dest` (default: `True`) |
| `overwrite` | `bool` | Overwrite an existing file at `dest` (default: `True`) |
| `db_range` | `int` | Dynamic range in dB (default: 90) |

Returns a `uint8` `np.ndarray` of shape `(height, width)` if `dest` is `None`, with row 0
the highest frequency; otherwise the path of the saved PNG. The values are indices into
`get_palette()`. Raises `ValueError` if the audio is too short for the width.

### `spectrogram_from_flac(flac_path, shape, dest=None, ...)`

Loads a FLAC file and makes a spectrogram in one call. Takes `shape`, `dest` and
`db_range` as `spectrogram()` does, plus:

| Argument | Type | Description |
|---|---|---|
| `flac_path` | `str` | Path to the FLAC file |
| `start_time` | `float`, optional | Start time in seconds |
| `segment` | `int`, optional | Overrides `start_time` with the `segment`-th (0-based) clip of `duration` |
| `duration` | `float` | Duration in seconds (default: 12) |
| `channel` | `int` | Channel to take (default: 0) |

### `load_audio(flac_path, start_time=None, segment=None, duration=None, channel=0)`

Reads a FLAC file. Returns `(samples, sample_rate)`, with the samples as a float64
`np.ndarray`. With no `duration` it reads to the end of the file.

### `get_palette()`

The palette of sox's grayscale spectrogram as a flat RGB list of 765 ints: 4 fixed
colours (background, text, labels, grid), then 251 grays from black to white. Index `i`
of `spectrogram()`'s array is the colour `palette[3*i : 3*i+3]`.

---

## AGREEMENT WITH SOX

`comparison/parity_check.sh -b 0.1.3 -n 20` renders 20 segments of each of the 10 dev
files (`s3://dse-soundhub/public/audio/dev/`) with `sox`, with sox_tensorflow 0.1.3, and
with sox_numpy, then runs PNW-Cnet v4 on each image. On darwin-arm64 (2026-11-19):

| | sox_tensorflow 0.1.3 | sox_numpy 0.1.0 |
|---|---|---|
| Mean exact-pixel match with `sox` | 99.8070% | 99.8075% |
| Segments with the same top class as on `sox`'s image | 199 of 200 | 199 of 200 |
| Mean of the largest class-probability difference | 0.000731 | 0.000730 |

sox_numpy is released only when it is no worse than 0.1.3 on all three. Its pixels
differ from 0.1.3's in about 0.002% of the test suite's reference pixels, always by one
level. Every segment is within ±2 levels of `sox`, and the remaining differences are in
the darkest pixels.

The algorithm, from SoX's C source:
- a Hann window with SoX's normalization, and SoX's partial windows at the start and
  end of the signal;
- a resample to 8 kHz with `soxr` (the SoX Resampler) at HQ quality, clipped to [-1, 1];
- SoX's dB conversion and its 251-level grayscale palette.

### Speed

CPU time per 12 s window at 48 kHz: 200 calls on one calling thread, after 5 warm-up calls (2026-11-19):

| | sox_tensorflow 0.1.3 (on `/CPU:0`) | sox_numpy 0.1.0 | sox_numpy, `OMP_NUM_THREADS=1` |
|---|---|---|---|
| darwin-arm64 (M5) | 14.3 ms | 10.3–11.3 ms | 3.5 ms |
| linux-x86_64 (under Rosetta) | 16.6 ms | 5.0 ms | 3.6 ms |

soxr resamples with OpenMP threads that use far more CPU than they save in wall time
(on the Mac, about 12 ms of CPU for 1.3 ms of wall). Set `OMP_NUM_THREADS=1` where
CPU is the limit; it does not change the pixels.

---

## DEVELOPMENT

Environments are managed with [pixi](https://pixi.sh), on osx-arm64, linux-64 and
linux-aarch64:

| Environment | For | Has |
|---|---|---|
| `default` | the package | sox_numpy |
| `test` | `pixi run -e test test` | pytest, Pillow; **no TensorFlow**, so a stray import fails the suite |
| `compare` | the [comparison harness](comparison/) | TensorFlow and tf_keras for PNW-Cnet, and conda-forge `sox` (not on linux-aarch64) |
| `dev` | releases | build, twine |

The tests compare every spectrogram with reference pixels for darwin-arm64 and
linux-x86_64 in `tests/fixtures/`. They change only when a change is meant to change
pixels; `tests/make_reference.py` remakes them.

When loading H5 models saved with older TensorFlow/Keras versions in `compare`, set
`export TF_USE_LEGACY_KERAS=1` (the parity check does).

---

## STYLE-GUIDE

Following PEP8. Keeping honest with `pycodestyle .`
