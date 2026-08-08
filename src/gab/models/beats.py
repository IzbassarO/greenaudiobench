"""BEATs iter3+ (AS2M) adapter — official microsoft/unilm implementation, vendored.

Official preprocessing (vendored BEATs.py lines 118-131, called via
extract_features): waveform * 2**15 -> kaldi fbank (128 mel bins, 16 kHz,
25 ms frames, 10 ms shift) -> (fbank - fbank_mean) / (2 * fbank_std) with the
checkpoint's stored constants. We call the official code, never reimplement.
Embedding = mean over time of the final encoder layer features (768-d);
fixed 5 s inputs mean there is no padding to mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..utils import REPO_ROOT, sha256_file
from ._beats_vendor.BEATs import BEATs, BEATsConfig
from .base import AdapterInfo, EmbeddingAdapter, freeze
from .registry import CHECKPOINTS

_CKPT = CHECKPOINTS["beats"]
_LOCAL = REPO_ROOT / "data" / "checkpoints" / "BEATs_iter3_plus_AS2M.pt"


def _checkpoint_path() -> Path:
    if not _LOCAL.exists():
        from huggingface_hub import hf_hub_download

        _LOCAL.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(
            repo_id="Bencr/beats-checkpoints", repo_type="dataset",
            filename="BEATs_iter3_plus_AS2M.pt", revision=_CKPT.revision,
        )
        Path(got).replace(_LOCAL) if Path(got).is_file() else None
    sha = sha256_file(_LOCAL)
    if sha != _CKPT.sha256:
        raise RuntimeError(
            f"beats checkpoint sha256 mismatch: got {sha}, pinned {_CKPT.sha256} "
            "(pin is cross-verified across three independent mirrors)"
        )
    return _LOCAL


class BeatsAdapter(EmbeddingAdapter):
    def __init__(self):
        self.info = AdapterInfo(
            name="beats",
            checkpoint=_CKPT.checkpoint,
            source_url=_CKPT.source_url,
            revision=_CKPT.revision,
            sample_rate=_CKPT.sample_rate,
            duration_policy="crop_or_pad:5.0s",
            preprocess_id="beats-official-preprocess(x*2^15;kaldi-fbank128;(x-mean)/(2*std))",
            embed_dim=768,
            embedding_layer="extract_features final layer, mean-pooled over time (768-d)",
        )
        self._model = None
        self._device = "cpu"
        self._dtype = torch.float32

    def load(self, device: str = "cpu", fp16: bool = False) -> None:
        ckpt = torch.load(_checkpoint_path(), map_location="cpu", weights_only=False)
        cfg = BEATsConfig(ckpt["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(ckpt["model"], strict=True)
        freeze(model)
        self._dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            model = model.half()
        self._model = model.to(device)
        self._device = device
        # keep the checkpoint's own fbank stats if stored in cfg (defaults match)
        self._fbank_mean = getattr(cfg, "fbank_mean", 15.41663)
        self._fbank_std = getattr(cfg, "fbank_std", 6.55582)

    def embed_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        batch = torch.from_numpy(np.stack(wavs)).to(self._device)
        if self._dtype == torch.float16:
            batch = batch.half()
        with torch.inference_mode():
            feats, _ = self._model.extract_features(
                batch, fbank_mean=self._fbank_mean, fbank_std=self._fbank_std
            )
            emb = feats.mean(dim=1)  # fixed-length input: no padding to mask
        return self.check_output(emb.float().cpu().numpy(), len(wavs))

    # --- efficiency hooks (M5) ---
    def torch_module(self):
        return self._model

    def example_tensor_input(self, batch_size: int = 1):
        x = torch.zeros(batch_size, 80000, dtype=torch.float32, device=self._device)
        return x, "waveform[B,80000]@16kHz (official fbank preprocess in path)"

    def _forward_tensor(self, x):
        feats, _ = self._model.extract_features(
            x, fbank_mean=self._fbank_mean, fbank_std=self._fbank_std)
        return feats.mean(dim=1)
