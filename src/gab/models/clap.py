"""LAION-CLAP adapter — laion/clap-htsat-unfused at the pinned revision.

Official preprocessing = the checkpoint's ClapFeatureExtractor (64 mel bands,
48 kHz, max_length_s=10). Determinism: the random-crop branch
(truncation="rand_trunc") is reachable ONLY for waveforms longer than
10 s (feature_extraction_clap.py: `if waveform.shape[0] > max_length`);
shorter input takes the deterministic repeat-pad path. The adapter asserts
input <= 10 s so the random branch is provably unreachable.
Embedding = ClapModel.get_audio_features -> projected, L2-normalized audio
embedding (modeling_clap.py applies F.normalize before returning), 512-d.
Also exposes embed_texts() (get_text_features, same 512-d space) for M4.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import AdapterInfo, EmbeddingAdapter, freeze
from .registry import CHECKPOINTS

_CKPT = CHECKPOINTS["laion_clap"]
_REPO_ID = "laion/clap-htsat-unfused"


class LaionClapAdapter(EmbeddingAdapter):
    def __init__(self):
        self.info = AdapterInfo(
            name="laion_clap",
            checkpoint=_CKPT.checkpoint,
            source_url=_CKPT.source_url,
            revision=_CKPT.revision,
            sample_rate=_CKPT.sample_rate,
            duration_policy="crop_or_pad:5.0s",
            preprocess_id="clap-feature-extractor(mel64;48k;repeatpad-to-10s)",
            embedding_layer="ClapModel.get_audio_features (projected, L2-normalized, 512-d)",
            embed_dim=512,
        )
        self._model = None
        self._processor = None
        self._device = "cpu"
        self._dtype = torch.float32

    def load(self, device: str = "cpu", fp16: bool = False) -> None:
        from transformers import ClapModel, ClapProcessor

        self._processor = ClapProcessor.from_pretrained(_REPO_ID, revision=_CKPT.revision)
        model = ClapModel.from_pretrained(_REPO_ID, revision=_CKPT.revision)
        freeze(model)
        self._dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            model = model.half()
        self._model = model.to(device)
        self._device = device

    def embed_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        fe = self._processor.feature_extractor
        max_samples = fe.max_length_s * fe.sampling_rate
        for w in wavs:
            if len(w) > max_samples:
                raise ValueError(
                    f"laion_clap: input {len(w)} samples > {max_samples} — the "
                    "rand_trunc branch would be nondeterministic; crop upstream"
                )
        inputs = fe([np.asarray(w, dtype=np.float32) for w in wavs],
                    sampling_rate=self.info.sample_rate, return_tensors="pt")
        x = inputs["input_features"].to(self._device, self._dtype)
        with torch.inference_mode():
            emb = self._model.get_audio_features(
                input_features=x, is_longer=inputs.get("is_longer").to(self._device)
                if inputs.get("is_longer") is not None else None,
            )
        return self.check_output(emb.float().cpu().numpy(), len(wavs))

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        tok = self._processor.tokenizer(texts, padding=True, return_tensors="pt")
        tok = {k: v.to(self._device) for k, v in tok.items()}
        with torch.inference_mode():
            emb = self._model.get_text_features(**tok)
        out = emb.float().cpu().numpy()
        if not np.isfinite(out).all():
            raise ValueError("laion_clap: NaN/Inf in text embeddings")
        return out

    # --- efficiency hooks (M5) ---
    def torch_module(self):
        return self._model.audio_model

    def example_tensor_input(self, batch_size: int = 1):
        fe = self._processor.feature_extractor
        feats = fe(np.zeros(5 * self.info.sample_rate, dtype=np.float32),
                   sampling_rate=self.info.sample_rate, return_tensors="pt")
        x = feats["input_features"].repeat(batch_size, 1, 1, 1).to(self._device, self._dtype)
        return x, f"mel{list(x.shape)} (FE output; FE itself is CPU-side, not in GMACs)"

    def _forward_tensor(self, x):
        return self._model.get_audio_features(input_features=x)
