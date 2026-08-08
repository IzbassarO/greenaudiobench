"""Canonical decode path (B2): deterministic, mono-mean, no per-clip normalization."""

import numpy as np
import pytest
import soundfile as sf

from gab.audio import fix_duration, load_audio


def write_wav(path, data, sr, subtype="PCM_16"):
    sf.write(str(path), data, sr, subtype=subtype)
    return path


def test_mono_mean_downmix(tmp_path):
    # L = 0.5, R = 0.0 -> mono mean must be 0.25 (uniform mean over channels)
    stereo = np.zeros((16000, 2), dtype=np.float32)
    stereo[:, 0] = 0.5
    wav = load_audio(write_wav(tmp_path / "s.wav", stereo, 16000), 16000)
    assert wav.ndim == 1 and wav.dtype == np.float32
    assert np.allclose(wav, 0.25, atol=2e-4)  # PCM16 quantization tolerance


def test_no_peak_normalization(tmp_path):
    # a quiet clip must STAY quiet after decode
    t = np.linspace(0, 1, 22050, endpoint=False)
    quiet = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = load_audio(write_wav(tmp_path / "q.wav", quiet, 22050), 22050)
    assert 0.09 < np.abs(wav).max() < 0.11


def test_resample_deterministic_and_correct_length(tmp_path):
    rng = np.random.default_rng(0)
    noise = (0.3 * rng.standard_normal(44100)).astype(np.float32)
    p = write_wav(tmp_path / "n.wav", noise, 44100)
    a = load_audio(p, 16000)
    b = load_audio(p, 16000)
    np.testing.assert_array_equal(a, b)  # bit-identical decode+resample
    assert len(a) == 16000  # 1 s at target rate
    assert np.isfinite(a).all()


def test_int16_range(tmp_path):
    full = np.array([32767, -32768] * 100, dtype=np.int16)
    wav = load_audio(write_wav(tmp_path / "f.wav", full, 8000, "PCM_16"), 8000)
    assert np.abs(wav).max() <= 1.0


def test_fix_duration_pad_and_crop():
    sr = 100
    short = np.ones(250, dtype=np.float32)
    padded = fix_duration(short, sr, 5.0)
    assert len(padded) == 500
    assert padded[:250].sum() == 250 and padded[250:].sum() == 0  # tail zero-pad
    long = np.arange(700, dtype=np.float32)
    cropped = fix_duration(long, sr, 5.0)
    assert len(cropped) == 500
    assert cropped[0] == 100  # center crop
    same = np.ones(500, dtype=np.float32)
    assert fix_duration(same, sr, 5.0) is same


@pytest.mark.parametrize("sr_in,sr_out", [(44100, 16000), (44100, 32000), (44100, 48000)])
def test_resample_preserves_tone_frequency(tmp_path, sr_in, sr_out):
    # a 440 Hz tone must still peak at 440 Hz after resampling
    t = np.linspace(0, 1, sr_in, endpoint=False)
    tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav = load_audio(write_wav(tmp_path / "t.wav", tone, sr_in), sr_out)
    spectrum = np.abs(np.fft.rfft(wav))
    peak_hz = np.fft.rfftfreq(len(wav), 1 / sr_out)[spectrum.argmax()]
    assert abs(peak_hz - 440) < 2
