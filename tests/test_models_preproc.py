"""B3 preprocessing sanity — offline, no checkpoint downloads.

Each test is designed to FAIL if a scaling/normalization step silently
disappears or a config constant drifts from the official values.
"""

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------- AST


def test_ast_feature_extractor_official_constants_and_shape():
    from transformers import ASTFeatureExtractor

    fe = ASTFeatureExtractor()  # constructor defaults == AudioSet checkpoint values
    assert fe.num_mel_bins == 128 and fe.max_length == 1024
    assert fe.do_normalize is True
    assert fe.mean == pytest.approx(-4.2677393)
    assert fe.std == pytest.approx(4.5689974)

    wav = (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 5, 80000))).astype(np.float32)
    out = fe(wav, sampling_rate=16000, return_tensors="np")["input_values"]
    assert out.shape == (1, 1024, 128)
    assert np.isfinite(out).all()


def test_ast_normalization_actually_applied():
    from transformers import ASTFeatureExtractor

    wav = (0.1 * np.random.default_rng(0).standard_normal(80000)).astype(np.float32)
    norm = ASTFeatureExtractor()(wav, sampling_rate=16000, return_tensors="np")["input_values"]
    raw = ASTFeatureExtractor(do_normalize=False)(
        wav, sampling_rate=16000, return_tensors="np")["input_values"]
    fe = ASTFeatureExtractor()
    # official formula: (x - mean) / (2 * std) — fails if constants/step change
    np.testing.assert_allclose(norm, (raw - fe.mean) / (2 * fe.std), atol=1e-4)


# ---------------------------------------------------------------- BEATs


def _beats_preprocess(wav_t, **kw):
    from gab.models._beats_vendor.BEATs import BEATs

    # preprocess touches no self attributes (vendored lines 118-131) — call unbound
    return BEATs.preprocess(None, wav_t, **kw)


def test_beats_official_scaling_2pow15_present():
    import torchaudio.compliance.kaldi as ta_kaldi

    wav = torch.from_numpy(
        (0.1 * np.sin(2 * np.pi * 440 * np.linspace(0, 5, 80000))).astype(np.float32)
    ).unsqueeze(0)
    official = _beats_preprocess(wav)
    # manual pipeline WITHOUT the 2**15 scaling: must differ massively
    fb_unscaled = ta_kaldi.fbank(wav[0].unsqueeze(0), num_mel_bins=128,
                                 sample_frequency=16000, frame_length=25,
                                 frame_shift=10)
    no_scale = (fb_unscaled - 15.41663) / (2 * 6.55582)
    diff = (official[0] - no_scale).abs().mean().item()
    assert diff > 1.0, "2**15 waveform scaling appears to be missing"


def test_beats_official_normalization_constants():
    import torchaudio.compliance.kaldi as ta_kaldi

    wav = torch.from_numpy(
        (0.1 * np.random.default_rng(1).standard_normal(80000)).astype(np.float32)
    ).unsqueeze(0)
    official = _beats_preprocess(wav)
    manual_fb = ta_kaldi.fbank((wav[0] * 2 ** 15).unsqueeze(0), num_mel_bins=128,
                               sample_frequency=16000, frame_length=25,
                               frame_shift=10)
    expected = (manual_fb - 15.41663) / (2 * 6.55582)
    torch.testing.assert_close(official[0], expected, atol=1e-5, rtol=1e-5)


def test_beats_preprocess_deterministic():
    wav = torch.from_numpy(
        (0.2 * np.random.default_rng(2).standard_normal(80000)).astype(np.float32)
    ).unsqueeze(0)
    a, b = _beats_preprocess(wav), _beats_preprocess(wav)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert a.shape[2] == 128 and torch.isfinite(a).all()


# ---------------------------------------------------------------- PANNs


def test_panns_cnn14_eval_deterministic_and_shapes():
    from gab.models._panns_vendor import Cnn14

    torch.manual_seed(0)
    m = Cnn14(sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64,
              fmin=50, fmax=14000, classes_num=527).eval()
    x = torch.from_numpy(
        (0.1 * np.random.default_rng(3).standard_normal((2, 160000))).astype(np.float32))
    with torch.inference_mode():
        a = m(x)["embedding"]
        b = m(x)["embedding"]
    torch.testing.assert_close(a, b, atol=0, rtol=0)  # SpecAug/dropout inactive
    assert a.shape == (2, 2048) and torch.isfinite(a).all()


def test_panns_official_mel_config():
    from gab.models._panns_vendor import Cnn14

    m = Cnn14(32000, 1024, 320, 64, 50, 14000, 527)
    assert m.logmel_extractor.melW.shape == (513, 64)  # n_fft 1024 -> 513 bins, 64 mels


# ------------------------------------------------------------- LAION-CLAP


def _clap_fe():
    from transformers import ClapFeatureExtractor

    # laion/clap-htsat-unfused preprocessor values
    return ClapFeatureExtractor(feature_size=64, sampling_rate=48000,
                                hop_length=480, max_length_s=10,
                                fft_window_size=1024)


def test_laion_clap_repeatpad_deterministic_and_tiles():
    # truncation="rand_trunc" is what the pinned checkpoint's preprocessor
    # config sets; the random branch is reachable only for > 10 s inputs.
    fe = _clap_fe()
    kw = dict(sampling_rate=48000, return_tensors="np", truncation="rand_trunc")
    wav = (0.1 * np.random.default_rng(4).standard_normal(240000)).astype(np.float32)
    a = fe(wav, **kw)
    b = fe(wav, **kw)
    np.testing.assert_array_equal(a["input_features"], b["input_features"])
    assert not np.asarray(a["is_longer"]).any()  # 5 s < 10 s: deterministic path
    assert a["input_features"].shape == (1, 1, 1001, 64)
    # repeatpad of a 5 s clip to 10 s == explicit 2x tile of the waveform
    tiled = fe(np.tile(wav, 2), **kw)
    np.testing.assert_allclose(a["input_features"], tiled["input_features"],
                               atol=1e-4)


# ---------------------------------------------------------------- MS-CLAP


def test_msclap_official_duration_pad_exact_tiling():
    from gab.models.msclap_adapter import official_duration_pad

    wav = (0.1 * np.random.default_rng(5).standard_normal(220500)).astype(np.float32)  # 5 s
    out = official_duration_pad(wav)
    assert out.shape == (308700,)  # exactly 7 s @ 44.1 kHz
    np.testing.assert_array_equal(out[:220500], wav)          # original first
    np.testing.assert_array_equal(out[220500:], wav[:88200])  # then repeat-tile
    np.testing.assert_array_equal(out, official_duration_pad(wav))  # deterministic


def test_msclap_refuses_overlong_input():
    from gab.models.msclap_adapter import official_duration_pad

    with pytest.raises(ValueError, match="nondeterministic"):
        official_duration_pad(np.zeros(308701, dtype=np.float32))
