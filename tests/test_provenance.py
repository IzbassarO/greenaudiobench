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


def test_cpu_cannot_masquerade_as_official_gpu(monkeypatch):
    monkeypatch.setattr(utils, "git_dirty", lambda: False)
    # on a machine without CUDA, an official GPU run must refuse to start
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present — guard not testable here")
    with pytest.raises(ProvenanceError, match="CUDA"):
        assert_official_run(require_cuda=True)
