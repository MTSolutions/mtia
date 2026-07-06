"""Qdrant collection management + search.

One collection per client (`docs_<client>`). Payload carries the
`blueprint_code` and `source` so we can filter by step type and delete by
document. Vector distance = Cosine (bge-m3 is trained for it).
"""

from __future__ import annotations

import hashlib
import os
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from modules.rag.storage import SHARED_CODE


VECTOR_SIZE = 1024  # bge-m3


def _qdrant_url() -> str:
    return os.environ.get("QDRANT_URL", "http://qdrant:6333")


def _client() -> QdrantClient:
    return QdrantClient(url=_qdrant_url())


def collection_name(client: str) -> str:
    if not client or "/" in client:
        raise ValueError(f"bad client: {client!r}")
    return f"docs_{client}"


def ensure_collection(client: str, dim: int = VECTOR_SIZE) -> None:
    qc = _client()
    name = collection_name(client)
    existing = {c.name for c in qc.get_collections().collections}
    if name in existing:
        return
    qc.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
    )
    # Payload indexes speed up filtered search + deletion.
    for field in ("blueprint_code", "source"):
        qc.create_payload_index(
            collection_name=name,
            field_name=field,
            field_schema=qm.PayloadSchemaType.KEYWORD,
        )


def drop_collection(client: str) -> None:
    qc = _client()
    name = collection_name(client)
    if name in {c.name for c in qc.get_collections().collections}:
        qc.delete_collection(collection_name=name)


@dataclass
class ChunkRecord:
    blueprint_code: str
    source: str
    page: int
    chunk_idx: int
    text: str
    embedding: list[float]


def _point_id(source: str, chunk_idx: int) -> str:
    # Deterministic UUID5 so re-ingesting the same source replaces same IDs.
    h = hashlib.sha1(f"{source}::{chunk_idx}".encode()).hexdigest()[:32]
    return str(uuid.UUID(h))


def upsert_chunks(client: str, chunks: list[ChunkRecord]) -> int:
    if not chunks:
        return 0
    ensure_collection(client, dim=len(chunks[0].embedding))
    qc = _client()
    points = [
        qm.PointStruct(
            id=_point_id(ch.source, ch.chunk_idx),
            vector=ch.embedding,
            payload={
                "blueprint_code": ch.blueprint_code,
                "source": ch.source,
                "page": ch.page,
                "chunk_idx": ch.chunk_idx,
                "text": ch.text,
            },
        )
        for ch in chunks
    ]
    qc.upsert(collection_name=collection_name(client), points=points, wait=True)
    return len(points)


def delete_by_source(client: str, blueprint_code: str, source: str) -> int:
    qc = _client()
    name = collection_name(client)
    if name not in {c.name for c in qc.get_collections().collections}:
        return 0
    flt = qm.Filter(must=[
        qm.FieldCondition(key="blueprint_code", match=qm.MatchValue(value=blueprint_code)),
        qm.FieldCondition(key="source", match=qm.MatchValue(value=source)),
    ])
    # count before delete, so callers can report progress
    pre = qc.count(collection_name=name, count_filter=flt, exact=True).count
    qc.delete(collection_name=name, points_selector=qm.FilterSelector(filter=flt), wait=True)
    return pre


def count_by_source(client: str, blueprint_code: str, source: str) -> int:
    qc = _client()
    name = collection_name(client)
    if name not in {c.name for c in qc.get_collections().collections}:
        return 0
    flt = qm.Filter(must=[
        qm.FieldCondition(key="blueprint_code", match=qm.MatchValue(value=blueprint_code)),
        qm.FieldCondition(key="source", match=qm.MatchValue(value=source)),
    ])
    return qc.count(collection_name=name, count_filter=flt, exact=True).count


@dataclass
class Hit:
    score: float
    source: str
    page: int
    chunk_idx: int
    text: str


def search(
    client: str,
    blueprint_code: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[Hit]:
    qc = _client()
    name = collection_name(client)
    if name not in {c.name for c in qc.get_collections().collections}:
        return []
    # Match either the step's blueprint or the shared bucket (OR semantics).
    flt = qm.Filter(should=[
        qm.FieldCondition(key="blueprint_code", match=qm.MatchValue(value=blueprint_code)),
        qm.FieldCondition(key="blueprint_code", match=qm.MatchValue(value=SHARED_CODE)),
    ])
    result = qc.query_points(
        collection_name=name,
        query=query_embedding,
        query_filter=flt,
        limit=top_k,
        with_payload=True,
    ).points
    return [
        Hit(
            score=p.score,
            source=p.payload.get("source", ""),
            page=p.payload.get("page", 0),
            chunk_idx=p.payload.get("chunk_idx", 0),
            text=p.payload.get("text", ""),
        )
        for p in result
    ]
