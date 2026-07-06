"""Integration tests for Qdrant retrieval. Requires Qdrant running
(docker compose -f docker-compose.yml -f docker-compose.mtia.yml up -d qdrant).
"""
from __future__ import annotations

import pytest

from modules.rag import retrieve, storage
from modules.rag.retrieve import ChunkRecord


def _fake_vec(seed: float, dim: int = 1024) -> list[float]:
    """Deterministic fake vector: a single non-zero component so cosine sims
    are clean (vectors with disjoint non-zero components have similarity 0)."""
    v = [0.0] * dim
    v[int(seed) % dim] = 1.0
    return v


@pytest.fixture
def collection(ephemeral_client):
    retrieve.ensure_collection(ephemeral_client, dim=1024)
    yield ephemeral_client
    retrieve.drop_collection(ephemeral_client)


def test_upsert_and_search_roundtrip(collection):
    chunks = [
        ChunkRecord("empaque", "manual.pdf", page=1, chunk_idx=0,
                    text="llenar cajas", embedding=_fake_vec(10)),
        ChunkRecord("empaque", "manual.pdf", page=2, chunk_idx=1,
                    text="etiquetar cajas", embedding=_fake_vec(20)),
        ChunkRecord("calidad", "qa.pdf", page=1, chunk_idx=0,
                    text="revisar peso", embedding=_fake_vec(30)),
    ]
    n = retrieve.upsert_chunks(collection, chunks)
    assert n == 3

    hits = retrieve.search(collection, "empaque", _fake_vec(10), top_k=5)
    sources = {h.source for h in hits}
    assert sources == {"manual.pdf"}
    assert hits[0].chunk_idx == 0


def test_search_filters_by_blueprint_code(collection):
    retrieve.upsert_chunks(collection, [
        ChunkRecord("empaque", "a.pdf", 1, 0, "x", _fake_vec(10)),
        ChunkRecord("calidad", "b.pdf", 1, 0, "y", _fake_vec(10)),
    ])
    empaque_hits = retrieve.search(collection, "empaque", _fake_vec(10), top_k=5)
    calidad_hits = retrieve.search(collection, "calidad", _fake_vec(10), top_k=5)
    assert [h.source for h in empaque_hits] == ["a.pdf"]
    assert [h.source for h in calidad_hits] == ["b.pdf"]


def test_client_isolation_uses_separate_collections(ephemeral_client):
    client_a = ephemeral_client
    client_b = ephemeral_client + "_b"
    try:
        retrieve.ensure_collection(client_a)
        retrieve.ensure_collection(client_b)
        retrieve.upsert_chunks(client_a, [
            ChunkRecord("empaque", "a.pdf", 1, 0, "a", _fake_vec(10)),
        ])
        retrieve.upsert_chunks(client_b, [
            ChunkRecord("empaque", "b.pdf", 1, 0, "b", _fake_vec(10)),
        ])
        assert [h.source for h in retrieve.search(client_a, "empaque", _fake_vec(10))] == ["a.pdf"]
        assert [h.source for h in retrieve.search(client_b, "empaque", _fake_vec(10))] == ["b.pdf"]
    finally:
        retrieve.drop_collection(client_a)
        retrieve.drop_collection(client_b)


def test_delete_by_source_removes_only_that_document(collection):
    retrieve.upsert_chunks(collection, [
        ChunkRecord("empaque", "a.pdf", 1, 0, "a0", _fake_vec(10)),
        ChunkRecord("empaque", "a.pdf", 2, 1, "a1", _fake_vec(11)),
        ChunkRecord("empaque", "b.pdf", 1, 0, "b0", _fake_vec(12)),
    ])
    removed = retrieve.delete_by_source(collection, "empaque", "a.pdf")
    assert removed == 2
    hits = retrieve.search(collection, "empaque", _fake_vec(10), top_k=10)
    assert {h.source for h in hits} == {"b.pdf"}


def test_search_includes_shared_chunks(collection):
    retrieve.upsert_chunks(collection, [
        ChunkRecord("despaletizadora_l1", "step.pdf", 1, 0,
                    "operación", _fake_vec(10)),
        ChunkRecord(storage.SHARED_CODE, "seguridad.pdf", 1, 0,
                    "riesgos eléctricos", _fake_vec(20)),
        ChunkRecord("other_step", "nope.pdf", 1, 0,
                    "irrelevante", _fake_vec(30)),
    ])
    hits = retrieve.search(collection, "despaletizadora_l1", _fake_vec(10), top_k=10)
    sources = {h.source for h in hits}
    assert sources == {"step.pdf", "seguridad.pdf"}


def test_upsert_is_idempotent_by_source_and_chunk_idx(collection):
    retrieve.upsert_chunks(collection, [
        ChunkRecord("empaque", "a.pdf", 1, 0, "old text", _fake_vec(10)),
    ])
    retrieve.upsert_chunks(collection, [
        ChunkRecord("empaque", "a.pdf", 1, 0, "new text", _fake_vec(10)),
    ])
    hits = retrieve.search(collection, "empaque", _fake_vec(10), top_k=5)
    assert len(hits) == 1
    assert hits[0].text == "new text"
