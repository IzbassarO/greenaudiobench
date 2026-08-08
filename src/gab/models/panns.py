"""PANNs CNN14 adapter — official checkpoint Cnn14_mAP=0.431.pth (Zenodo).

Official config for this checkpoint (audioset_tagging_cnn README/pytorch/main.py):
sample_rate=32000, window_size=1024, hop_size=320, mel_bins=64, fmin=50,
fmax=14000, classes_num=527. The log-mel frontend is torchlibrosa INSIDE
Cnn14.forward, so the waveform->embedding latency region includes it.
Embedding = the 2048-d penultimate 'embedding' output (relu(fc1); dropout is
inactive in eval mode, SpecAugment/mixup are training-only paths).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..utils import REPO_ROOT, sha256_file
from ._panns_vendor import Cnn14
from .base import AdapterInfo, EmbeddingAdapter, freeze
from .registry import CHECKPOINTS

_CKPT = CHECKPOINTS["panns_cnn14"]
_LOCAL = REPO_ROOT / "data" / "checkpoints" / "Cnn14_mAP=0.431.pth"
_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def _checkpoint_path() -> Path:
    if not _LOCAL.exists():
        import requests

        _LOCAL.parent.mkdir(parents=True, exist_ok=True)
        tmp = _LOCAL.with_suffix(".pth.part")
        with requests.get(_URL, stream=True, timeout=(30, 300)) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        tmp.rename(_LOCAL)
    sha = sha256_file(_LOCAL)
    if sha != _CKPT.sha256:
        raise RuntimeError(
            f"panns_cnn14 checkpoint sha256 mismatch: got {sha}, pinned {_CKPT.sha256}"
        )
    return _LOCAL


class PannsCnn14Adapter(EmbeddingAdapter):
    def __init__(self):
        self.info = AdapterInfo(
            name="panns_cnn14",
            checkpoint=_CKPT.checkpoint,
            source_url=_CKPT.source_url,
            revision=_CKPT.revision,
            sample_rate=_CKPT.sample_rate,
            duration_policy="crop_or_pad:5.0s",
            preprocess_id="panns-cnn14-logmel64-win1024-hop320-fmin50-fmax14000(in-model)",
            embed_dim=2048,
            embedding_layer="Cnn14 forward 'embedding' (relu(fc1), 2048-d, eval mode)",
        )
        self._model = None
        self._device = "cpu"
        self._dtype = torch.float32

    def load(self, device: str = "cpu", fp16: bool = False) -> None:
        model = Cnn14(sample_rate=32000, window_size=1024, hop_size=320,
                      mel_bins=64, fmin=50, fmax=14000, classes_num=527)
        state = torch.load(_checkpoint_path(), map_location="cpu", weights_only=True)
        model.load_state_dict(state["model"], strict=True)
        freeze(model)
        self._dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            model = model.half()
        self._model = model.to(device)
        self._device = device

    def embed_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        batch = torch.from_numpy(np.stack(wavs)).to(self._device, self._dtype)
        with torch.inference_mode():
            out = self._model(batch)["embedding"]
        return self.check_output(out.float().cpu().numpy(), len(wavs))

    # --- efficiency hooks (M5) ---
    def torch_module(self):
        return self._model

    def example_tensor_input(self, batch_size: int = 1):
        x = torch.zeros(batch_size, 160000, dtype=self._dtype, device=self._device)
        return x, "waveform[B,160000]@32kHz (log-mel frontend in-graph)"

    def _forward_tensor(self, x):
        return self._model(x)["embedding"]
