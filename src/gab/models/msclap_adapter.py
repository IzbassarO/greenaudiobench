"""MS-CLAP 2023 adapter — CLAP_weights_2023.pth from HF microsoft/msclap, pinned.

B2 compliance: waveforms in, never file paths. We replicate the package's
load_audio_into_tensor duration handling (msclap/CLAPWrapper.py lines
227-248) EXACTLY for the <= duration case: repeat-tile by ceil factor, then
truncate to duration*sr (deterministic). The random-crop branch exists only
for clips LONGER than the 7 s config duration (config_2023.yml: duration=7,
sampling_rate=44100); the adapter asserts input <= 7 s so it is unreachable.
Encoding then follows _get_audio_embeddings: clap.audio_encoder(batch)[0]
(projected audio embedding, 1024-d). Weights are downloaded at the pinned
revision and sha256-verified against the registry BEFORE use.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch

from ..utils import REPO_ROOT, sha256_file
from .base import AdapterInfo, EmbeddingAdapter, freeze
from .registry import CHECKPOINTS

_CKPT = CHECKPOINTS["ms_clap"]
_LOCAL = REPO_ROOT / "data" / "checkpoints" / "CLAP_weights_2023.pth"
_DURATION_S = 7          # config_2023.yml: duration
_SR = 44100              # config_2023.yml: sampling_rate


def _weights_path() -> Path:
    if not _LOCAL.exists():
        from huggingface_hub import hf_hub_download

        _LOCAL.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(repo_id="microsoft/msclap",
                              filename="CLAP_weights_2023.pth",
                              revision=_CKPT.revision)
        _LOCAL.write_bytes(Path(got).read_bytes())
    sha = sha256_file(_LOCAL)
    if sha != _CKPT.sha256:
        raise RuntimeError(
            f"ms_clap weights sha256 mismatch: got {sha}, pinned {_CKPT.sha256}"
        )
    return _LOCAL


def official_duration_pad(wav: np.ndarray, sr: int = _SR,
                          duration_s: int = _DURATION_S) -> np.ndarray:
    """msclap load_audio_into_tensor semantics for clips <= duration (exact)."""
    n_target = duration_s * sr
    n = wav.shape[0]
    if n > n_target:
        raise ValueError(
            f"ms_clap: input {n} samples > {n_target} — the random-crop branch "
            "would be nondeterministic; crop upstream"
        )
    repeat_factor = int(math.ceil(n_target / n))
    tiled = np.tile(wav, repeat_factor)
    return tiled[:n_target]


class MsClapAdapter(EmbeddingAdapter):
    def __init__(self):
        self.info = AdapterInfo(
            name="ms_clap",
            checkpoint=_CKPT.checkpoint,
            source_url=_CKPT.source_url,
            revision=_CKPT.revision,
            sample_rate=_CKPT.sample_rate,
            duration_policy="crop_or_pad:5.0s",
            preprocess_id="msclap2023-official(repeat-tile-to-7s;encoder-internal-mel)",
            embedding_layer="clap.audio_encoder(...)[0] projected embedding (1024-d)",
            embed_dim=1024,
        )
        self._wrapper = None
        self._device = "cpu"
        self._dtype = torch.float32

    def load(self, device: str = "cpu", fp16: bool = False) -> None:
        from msclap import CLAP

        use_cuda = device == "cuda"
        self._wrapper = CLAP(model_fp=str(_weights_path()), version="2023",
                             use_cuda=use_cuda)
        freeze(self._wrapper.clap)
        self._dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            self._wrapper.clap.half()
        self._device = device

    def embed_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        batch = np.stack([official_duration_pad(np.asarray(w, np.float32))
                          for w in wavs])
        x = torch.from_numpy(batch).to(self._device, self._dtype)
        with torch.inference_mode():
            emb = self._wrapper.clap.audio_encoder(x)[0]
        return self.check_output(emb.float().cpu().numpy(), len(wavs))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        with torch.inference_mode():
            emb = self._wrapper.get_text_embeddings(texts)
        out = emb.float().cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError("ms_clap: NaN/Inf in text embeddings")
        return out

    # --- efficiency hooks (M5) ---
    def torch_module(self):
        return self._wrapper.clap.audio_encoder

    def example_tensor_input(self, batch_size: int = 1):
        x = torch.zeros(batch_size, _DURATION_S * _SR, dtype=self._dtype,
                        device=self._device)
        return x, f"waveform[B,{_DURATION_S * _SR}]@44.1kHz (encoder-internal mel in-graph)"

    def _forward_tensor(self, x):
        return self._wrapper.clap.audio_encoder(x)[0]
