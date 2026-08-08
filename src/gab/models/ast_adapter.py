"""AST adapter — MIT/ast-finetuned-audioset-10-10-0.4593 at the pinned revision.

Official preprocessing = the checkpoint's own ASTFeatureExtractor
(transformers feature_extraction_audio_spectrogram_transformer.py): kaldi
fbank 128 mel bins, pad/truncate to max_length=1024 frames (5 s @16 kHz ->
512 frames, zero-padded to 1024), AudioSet normalization (x - mean)/(2*std)
with the checkpoint's stored mean/std. Embedding = ASTModel pooled output:
(sequence_output[:,0] + sequence_output[:,1]) / 2 after final LayerNorm —
mean of CLS + distillation tokens (modeling_...py line 404), 768-d.
"""

from __future__ import annotations

import numpy as np
import torch

from .base import AdapterInfo, EmbeddingAdapter, freeze
from .registry import CHECKPOINTS

_CKPT = CHECKPOINTS["ast"]
_REPO_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"


class ASTAdapter(EmbeddingAdapter):
    def __init__(self):
        self.info = AdapterInfo(
            name="ast",
            checkpoint=_CKPT.checkpoint,
            source_url=_CKPT.source_url,
            revision=_CKPT.revision,
            sample_rate=_CKPT.sample_rate,
            duration_policy="crop_or_pad:5.0s",
            preprocess_id="ast-feature-extractor(fbank128;maxlen1024;audioset-norm)",
            embedding_layer="ASTModel pooled output ((cls+distill)/2 after LayerNorm, 768-d)",
            embed_dim=768,
        )
        self._model = None
        self._fe = None
        self._device = "cpu"
        self._dtype = torch.float32

    def load(self, device: str = "cpu", fp16: bool = False) -> None:
        from transformers import ASTFeatureExtractor, ASTModel

        self._fe = ASTFeatureExtractor.from_pretrained(_REPO_ID, revision=_CKPT.revision)
        model = ASTModel.from_pretrained(_REPO_ID, revision=_CKPT.revision)
        freeze(model)
        self._dtype = torch.float16 if fp16 else torch.float32
        if fp16:
            model = model.half()
        self._model = model.to(device)
        self._device = device

    def embed_batch(self, wavs: list[np.ndarray]) -> np.ndarray:
        # feature extraction INSIDE the timed path (fair latency region)
        inputs = self._fe(
            [np.asarray(w, dtype=np.float32) for w in wavs],
            sampling_rate=self.info.sample_rate, return_tensors="pt",
        )
        x = inputs["input_values"].to(self._device, self._dtype)
        with torch.inference_mode():
            out = self._model(input_values=x)
        return self.check_output(out.pooler_output.float().cpu().numpy(), len(wavs))

    # --- efficiency hooks (M5) ---
    def torch_module(self):
        return self._model

    def example_tensor_input(self, batch_size: int = 1):
        x = torch.zeros(batch_size, self._fe.max_length, self._fe.num_mel_bins,
                        dtype=self._dtype, device=self._device)
        return x, f"fbank[B,{self._fe.max_length},{self._fe.num_mel_bins}] (FE output; FE itself is CPU-side, not in GMACs)"

    def _forward_tensor(self, x):
        return self._model(input_values=x).pooler_output
