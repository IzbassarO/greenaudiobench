"""Common adapter interface: canonical waveform -> frozen encoder -> embedding.

Every model adapter:
- receives mono float32 waveforms ALREADY decoded by gab.audio.load_audio at
  the adapter's declared sample_rate (adapters never open files);
- applies its OWN official/reference preprocessing (feature extraction is
  model-specific and happens INSIDE the adapter, so the fair latency region
  waveform->embedding covers it);
- runs a frozen backbone: eval(), requires_grad_(False), torch.inference_mode();
- returns one fixed-size embedding per clip.
"""

from __future__ import annotations

import abc
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class AdapterInfo:
    """Identity + preprocessing metadata stored with every embedding cache."""

    name: str               # registry key, e.g. "ast"
    checkpoint: str         # exact checkpoint id/filename
    source_url: str         # official source (HF repo / Zenodo / MS release)
    revision: str           # HF commit sha, git commit, or file sha256
    sample_rate: int        # Hz expected at embed_batch input
    duration_policy: str    # e.g. "crop_or_pad:5.0s" (gab.audio.fix_duration)
    preprocess_id: str      # official feature-extraction identifier
    embed_dim: int
    embedding_layer: str    # which output is used as the embedding

    def as_dict(self) -> dict:
        return asdict(self)


class EmbeddingAdapter(abc.ABC):
    """Lifecycle: adapter = X(); adapter.load(device, fp16=False); adapter.embed_batch(...)."""

    info: AdapterInfo

    @abc.abstractmethod
    def load(self, device: str = "cpu", fp16: bool = False) -> None:
        """Instantiate the FROZEN model on device. Must set eval() and disable grads."""

    @abc.abstractmethod
    def embed_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        """Embed a batch of canonical waveforms (each mono float32 at
        info.sample_rate, duration already fixed per info.duration_policy).
        Returns float32 (len(wavs), info.embed_dim). Must be finite."""

    def check_output(self, emb: np.ndarray, n_expected: int) -> np.ndarray:
        emb = np.asarray(emb, dtype=np.float32)
        if emb.shape != (n_expected, self.info.embed_dim):
            raise ValueError(
                f"{self.info.name}: embedding shape {emb.shape} != "
                f"({n_expected}, {self.info.embed_dim})"
            )
        if not np.isfinite(emb).all():
            raise ValueError(f"{self.info.name}: NaN/Inf in embeddings")
        return emb


def freeze(module) -> None:
    """Shared frozen-backbone enforcement for all adapters."""
    module.eval()
    for p in module.parameters():
        p.requires_grad_(False)
