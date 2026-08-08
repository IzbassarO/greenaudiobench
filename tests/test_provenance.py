"""B1: dirty-tree guard and mandatory provenance columns."""

import pytest

from gab import utils
from gab.utils import PROVENANCE_COLUMNS, ProvenanceError, assert_official_run, run_metadata


def test_run_metadata_has_all_mandatory_columns():
    meta = run_metadata()
    assert set(PROVENANCE_COLUMNS) <= set(meta)
    assert meta["git_commit"] not in ("", None)
    assert isinstance(meta["git_dirty"], bool)


def test_official_run_refuses_dirty_tree(monkeypatch):
    monkeypatch.setattr(utils, "git_dirty", lambda: True)
    with pytest.raises(ProvenanceError, match="dirty"):
        assert_official_run(require_cuda=False, allow_dirty=False)


def test_official_run_allows_explicit_debug_override(monkeypatch):
    monkeypatch.setattr(utils, "git_dirty", lambda: True)
    meta = assert_official_run(require_cuda=False, allow_dirty=True)
    assert meta["git_dirty"] is True  # override never hides dirtiness in the record


def test_git_dirty_excludes_experiment_outputs(monkeypatch):
    """Own outputs (results/, figures/) must not poison the clean-tree guard,
    while source changes must still count as dirty."""
    calls = {}

    def fake_git(*args):
        calls["args"] = args
        return ""  # git reports nothing outside the excluded dirs

    monkeypatch.setattr(utils, "_git", fake_git)
    assert utils.git_dirty() is False
    assert ":(exclude)results" in calls["args"]
    assert ":(exclude)figures" in calls["args"]

    monkeypatch.setattr(utils, "_git", lambda *a: " M src/gab/utils.py")
    assert utils.git_dirty() is True  # source modification still flags dirty


def test_beats_checkpoint_fetch_copies_bytes(monkeypatch, tmp_path):
    """hf_hub_download may return a relative symlink into the HF cache; the
    adapter must byte-copy it, never move the symlink itself."""
    from gab.models import beats as beats_mod

    blob = tmp_path / "blobs" / "abc"
    blob.parent.mkdir()
    blob.write_bytes(b"checkpoint-bytes")
    link = tmp_path / "BEATs_iter3_plus_AS2M.pt"
    link.symlink_to("blobs/abc")  # relative symlink, like the HF cache layout

    local = tmp_path / "checkpoints" / "BEATs_iter3_plus_AS2M.pt"
    local.parent.mkdir()
    monkeypatch.setattr(beats_mod, "_LOCAL", local)
    monkeypatch.setattr(beats_mod, "sha256_file", lambda p: beats_mod._CKPT.sha256)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda **kw: str(link))
    assert beats_mod._checkpoint_path() == local
    assert not local.is_symlink()
    assert local.read_bytes() == b"checkpoint-bytes"  # real bytes, target intact
    assert blob.exists()


def test_cpu_cannot_masquerade_as_official_gpu(monkeypatch):
    monkeypatch.setattr(utils, "git_dirty", lambda: False)
    # on a machine without CUDA, an official GPU run must refuse to start
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present — guard not testable here")
    with pytest.raises(ProvenanceError, match="CUDA"):
        assert_official_run(require_cuda=True)
