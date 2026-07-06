"""Unit tests for filesystem storage helpers."""
from __future__ import annotations

import pytest

from modules.rag import storage


def test_list_documents_returns_empty_when_missing():
    assert storage.list_documents("acme", "empaque") == []


def test_save_upload_and_list(tmp_path):
    storage.save_upload("acme", "empaque", "manual.pdf", b"hello")
    docs = storage.list_documents("acme", "empaque")
    assert len(docs) == 1
    assert docs[0].source == "manual.pdf"
    assert docs[0].size_bytes == 5


def test_unsafe_client_name_rejected():
    with pytest.raises(ValueError):
        storage.list_documents("acme/../etc", "empaque")


def test_unsafe_source_rejected():
    with pytest.raises(ValueError):
        storage.resolve_source("acme", "empaque", "../evil.pdf")


def test_delete_source_removes_file():
    storage.save_upload("acme", "empaque", "m.pdf", b"x")
    assert storage.delete_source("acme", "empaque", "m.pdf") is True
    assert storage.list_documents("acme", "empaque") == []
    assert storage.delete_source("acme", "empaque", "m.pdf") is False


def test_list_clients_and_blueprints():
    storage.save_upload("acme", "empaque", "a.pdf", b"x")
    storage.save_upload("acme", "control", "b.pdf", b"x")
    storage.save_upload("beta", "empaque", "c.pdf", b"x")
    assert storage.list_clients() == ["acme", "beta"]
    assert storage.list_blueprints("acme") == ["control", "empaque"]


def test_list_shared_documents_returns_dir_contents():
    storage.save_upload("acme", "empaque", "step.pdf", b"x")
    storage.save_upload("acme", storage.SHARED_CODE, "seguridad.pdf", b"xx")
    shared = storage.list_shared_documents("acme")
    assert [d.source for d in shared] == ["seguridad.pdf"]
    assert shared[0].size_bytes == 2
