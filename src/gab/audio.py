"""Canonical audio decode path — the single entry point for ALL model wrappers.

Contract (audit blocker B2):
    load_audio(path, target_sr) -> mono float32 numpy waveform in [-1, 1]

- deterministic decode via libsndfile (soundfile), no randomness anywhere;
- explicit channel handling: multi-channel -> mono by uniform mean over channels;
- explicit resampling: torchaudio sinc interpolation (kaiser window) at a fixed,
  documented parameterization — identical results on CPU and GPU/Colab;
- NO per-clip peak/RMS normalization: soundfile already yields float32 in
  [-1, 1] for integer PCM (scaled by full-scale), and model wrappers must apply
  their OWN official preprocessing after this canonical decode.

Model wrappers receive waveform arrays from here — never file paths.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

#: Identifier stored in embedding-cache metadata so cached artifacts can prove
#: which decode produced them. Bump when ANY behavior in this module changes.
DECODE_ID = "gab-decode-v1:soundfile-float32-mono-mean/torchaudio-resample-sinc-kaiser"

_RESAMPLE_KWARGS = dict(
    resampling_method="sinc_interp_kaiser",
    lowpass_filter_width=64,
    rolloff=0.9475937167399596,
    beta=14.769656459379492,
)


def load_audio(path: str | Path, target_sr: int) -> np.ndarray:
    """Decode one clip to a mono float32 waveform at target_sr. See module doc."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    # explicit channel -> mono: uniform mean across channels
    wav = data.mean(axis=1)
    if sr != target_sr:
        wav = _resample(wav, sr, target_sr)
    return np.ascontiguousarray(wav, dtype=np.float32)


def _resample(wav: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    import torch  # local import: keeps soundfile-only callers torch-free
    import torchaudio.functional as taf

    with torch.no_grad():
        out = taf.resample(
            torch.from_numpy(wav).unsqueeze(0), sr, target_sr, **_RESAMPLE_KWARGS
        )
    return out.squeeze(0).numpy()


def fix_duration(wav: np.ndarray, target_sr: int, seconds: float) -> np.ndarray:
    """Deterministic duration policy: center-crop if longer, zero-pad tail if shorter.

    ESC-50 clips are exactly 5 s so this is a no-op there; it exists so every
    model adapter states its duration policy explicitly (recorded in cache
    metadata as e.g. "crop_or_pad:5.0s").
    """
    n_target = int(round(seconds * target_sr))
    n = len(wav)
    if n == n_target:
        return wav
    if n > n_target:
        start = (n - n_target) // 2
        return wav[start:start + n_target]
    out = np.zeros(n_target, dtype=np.float32)
    out[:n] = wav
    return out
