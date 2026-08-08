"""Pinned checkpoint identities (audit blocker C3) — single source of truth.

Every field here is recorded into embedding-cache metadata and MODELS.md.
`revision` / `sha256` values marked PENDING are filled the moment the exact
pin is resolved from the official source — extraction REFUSES to run in
official mode while any selected model is still PENDING.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Checkpoint:
    name: str            # registry key (also cache dir name)
    checkpoint: str      # exact file / HF repo id
    source_url: str      # official download page / repo
    revision: str        # HF commit sha or upstream git commit ("PENDING" until pinned)
    sha256: str | None   # of the checkpoint file, when a direct file exists
    sample_rate: int     # canonical decode target for this model
    adapter: str         # "gab.models.<module>:<Class>"


CHECKPOINTS: dict[str, Checkpoint] = {
    # Pinned 2026-08-08. HF sha256 values are the LFS checksums reported by the
    # Hugging Face API for the pinned revision; the hub client re-verifies them
    # at download time.
    "ast": Checkpoint(
        name="ast",
        checkpoint="MIT/ast-finetuned-audioset-10-10-0.4593 (model.safetensors)",
        source_url="https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593",
        revision="f826b80d28226b62986cc218e5cec390b1096902",
        sha256="ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d",
        sample_rate=16000,
        adapter="gab.models.ast_adapter:ASTAdapter",
    ),
    "laion_clap": Checkpoint(
        name="laion_clap",
        checkpoint="laion/clap-htsat-unfused (pytorch_model.bin)",
        source_url="https://huggingface.co/laion/clap-htsat-unfused",
        revision="8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a",
        sha256="1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb",
        sample_rate=48000,
        adapter="gab.models.clap:LaionClapAdapter",
    ),
    "panns_cnn14": Checkpoint(
        name="panns_cnn14",
        checkpoint="Cnn14_mAP=0.431.pth",
        source_url="https://zenodo.org/records/3987831",
        revision="zenodo-3987831 (md5 541141fa2ee191a88f24a3219fff024e)",
        sha256="PENDING_LOCAL_VERIFY",  # computed from the downloaded file
        sample_rate=32000,
        adapter="gab.models.panns:PannsCnn14Adapter",
    ),
    # Official microsoft/unilm BEATs links are DEAD as of 2026-08-08 (unilm
    # issue #1671 open since 2024). Pinned to the most-used HF mirror; the
    # file sha256 is byte-identical across three independent mirrors.
    "beats": Checkpoint(
        name="beats",
        checkpoint="BEATs_iter3_plus_AS2M.pt (self-supervised iter3+ AS2M, 361,499,833 bytes)",
        source_url="https://huggingface.co/datasets/Bencr/beats-checkpoints"
                   " (mirror; official: github.com/microsoft/unilm/tree/master/beats)",
        revision="082fb1849d55bef1ee52ee8d8910b3adc69d4bc8",
        sha256="d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34",
        sample_rate=16000,
        adapter="gab.models.beats:BeatsAdapter",
    ),
    "ms_clap": Checkpoint(
        name="ms_clap",
        checkpoint="CLAP_weights_2023.pth (microsoft/msclap)",
        source_url="https://huggingface.co/microsoft/msclap",
        revision="c47d441165daa21986ead0850660917636a81775",
        sha256="2cef4016d47d00eb28d153d522f397222057f95000e9bad6b9583c631284a1e6",
        sample_rate=44100,
        adapter="gab.models.msclap_adapter:MsClapAdapter",
    ),
}

#: Extraction order fixed by protocol (do not reorder, do not extend).
MODEL_ORDER = ("ast", "laion_clap", "panns_cnn14", "beats", "ms_clap")


def load_adapter(name: str):
    """Instantiate the adapter class for a registry entry."""
    import importlib

    module_name, class_name = CHECKPOINTS[name].adapter.split(":")
    return getattr(importlib.import_module(module_name), class_name)()


def assert_pinned(names: list[str]) -> None:
    pending = [n for n in names if CHECKPOINTS[n].revision == "PENDING"]
    if pending:
        raise RuntimeError(
            f"checkpoints not pinned yet for official run: {pending} "
            "(fill revision/sha256 in gab/models/registry.py first)"
        )
