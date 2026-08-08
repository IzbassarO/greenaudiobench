# Model checkpoints — pinned identities (audit blocker C3)

Machine-readable source of truth: `src/gab/models/registry.py`. Extraction
refuses to run in official mode unless every selected model is pinned here.

Pinned 2026-08-08. HF sha256 = LFS checksum reported by the Hugging Face API
for the pinned revision (the hub client re-verifies file checksums at
download time). Direct-file checkpoints (PANNs, BEATs) are additionally
verified against sha256 computed from the actually-downloaded file.

| Model | Checkpoint file | Official source | Pinned revision | Size (bytes) | SHA-256 | Input SR | Embedding |
|---|---|---|---|---|---|---|---|
| AST | `model.safetensors` of `MIT/ast-finetuned-audioset-10-10-0.4593` | HF MIT/ast-finetuned-audioset-10-10-0.4593 | `f826b80d28226b62986cc218e5cec390b1096902` | 346,404,948 | `ae0c1e2ad4e1381d851fa9bf298ba13ebc9c5a914cdee2dbe427a6583869924d` | 16 kHz | pooled encoder output, 768-d |
| LAION-CLAP | `pytorch_model.bin` of `laion/clap-htsat-unfused` | HF laion/clap-htsat-unfused | `8fa0f1c6d0433df6e97c127f64b2a1d6c0dcda8a` | 614,525,833 | `1cd3c601bc4afe0fa87be3de4c13dd2cfadd249fac1e29acf74a9b296c3219bb` | 48 kHz | projected audio embedding, 512-d |
| PANNs CNN14 | `Cnn14_mAP=0.431.pth` | Zenodo 10.5281/zenodo.3987831 (qiuqiangkong/audioset_tagging_cnn) | zenodo-3987831; md5 `541141fa2ee191a88f24a3219fff024e` | 327,428,481 | recorded from downloaded file (see registry) | 32 kHz | penultimate `embedding` output, 2048-d |
| BEATs iter3+ AS2M | `BEATs_iter3_plus_AS2M.pt` (self-supervised, NOT the finetuned cpt1/cpt2) | **official link dead** (unilm issue #1671); mirror HF `Bencr/beats-checkpoints` | `082fb1849d55bef1ee52ee8d8910b3adc69d4bc8` | 361,499,833 | `d43cbfad4d7b56381c061d7a24774f908d4d94c72961f6eb1d9090ff18cd8d34` | 16 kHz | time-pooled encoder features, 768-d |
| MS-CLAP 2023 | `CLAP_weights_2023.pth` of `microsoft/msclap` | HF microsoft/msclap (github.com/microsoft/CLAP) | `c47d441165daa21986ead0850660917636a81775` | 689,950,036 | `2cef4016d47d00eb28d153d522f397222057f95000e9bad6b9583c631284a1e6` | 44.1 kHz | projected audio embedding, 1024-d |

## BEATs provenance note (honest disclosure)

The official checkpoint links on the microsoft/unilm BEATs page (OneDrive)
are dead as of 2026-08-08 — microsoft/unilm issue #1671 remains open with
user reports through 2025-12. We therefore pin the most-downloaded Hugging
Face mirror of `BEATs_iter3_plus_AS2M.pt`. The file's size and LFS sha256 are
byte-identical across three independent mirrors (`Bencr/beats-checkpoints`,
`lpepino/beats_ckpts`, `mooneyko/BEATs`), which is the strongest available
integrity evidence absent a live official link. This will be disclosed in the
paper's reproducibility statement.

## Dataset source revisions

- ESC-50: pinned to commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`
  (verified: this sha is embedded as the zip archive comment of the actually
  downloaded `esc50-master.zip`; future downloads use the pinned-commit URL).
- UrbanSound8K: Zenodo DOI 10.5281/zenodo.1203745 (versioned record).
