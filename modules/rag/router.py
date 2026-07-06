"""FastAPI router for /rag/* endpoints."""
from __future__ import annotations

import json
from typing import AsyncIterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sse_starlette.sse import EventSourceResponse

from modules.rag import embeddings, ingest, llm, retrieve, storage
from modules.rag.auth import JwtClaims, require_client_match, verify_jwt
from modules.rag.schemas import (
    Citation,
    DeleteResponse,
    DocumentInfo,
    DocumentListResponse,
    IngestResponse,
)


router = APIRouter(prefix="/rag", tags=["rag"])


# Retrieval score threshold below which we refuse to answer.
# Cosine similarity with bge-m3 — loose floor for the MVP; tune later.
MIN_RETRIEVAL_SCORE = 0.30


@router.get("/documents", response_model=DocumentListResponse)
def list_documents(
    client: str,
    blueprint_code: str,
    claims: JwtClaims = Depends(verify_jwt),
) -> DocumentListResponse:
    require_client_match(client, claims)

    step_docs = storage.list_documents(client, blueprint_code)
    documents = [
        DocumentInfo(
            source=d.source,
            size_bytes=d.size_bytes,
            mtime=d.mtime,
            chunks=retrieve.count_by_source(client, blueprint_code, d.source),
            blueprint_code=blueprint_code,
        )
        for d in step_docs
    ]
    # Fold in shared docs unless the caller is already asking for them.
    # If a filename exists in both buckets, the step-specific copy wins.
    if blueprint_code != storage.SHARED_CODE:
        step_sources = {d.source for d in step_docs}
        for d in storage.list_shared_documents(client):
            if d.source in step_sources:
                continue
            documents.append(DocumentInfo(
                source=d.source,
                size_bytes=d.size_bytes,
                mtime=d.mtime,
                chunks=retrieve.count_by_source(client, storage.SHARED_CODE, d.source),
                blueprint_code=storage.SHARED_CODE,
            ))
    return DocumentListResponse(
        client=client, blueprint_code=blueprint_code, documents=documents)


@router.post("/ingest", response_model=IngestResponse)
async def ingest_upload(
    client: str = Form(...),
    blueprint_code: str = Form(...),
    file: UploadFile = File(...),
    claims: JwtClaims = Depends(verify_jwt),
) -> IngestResponse:
    require_client_match(client, claims)
    if not file.filename:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "file has no filename")

    data = await file.read()
    path = storage.save_upload(client, blueprint_code, file.filename, data)
    stats = await ingest.ingest_file(client, blueprint_code, path)
    return IngestResponse(**{k: stats[k] for k in ("source", "chunks", "pages", "seconds")})


@router.delete("/documents", response_model=DeleteResponse)
def delete_document(
    client: str,
    blueprint_code: str,
    source: str,
    claims: JwtClaims = Depends(verify_jwt),
) -> DeleteResponse:
    require_client_match(client, claims)
    storage.delete_source(client, blueprint_code, source)
    removed = retrieve.delete_by_source(client, blueprint_code, source)
    return DeleteResponse(source=source, deleted_chunks=removed)


@router.get("/documents/content")
def document_content(
    client: str,
    blueprint_code: str,
    source: str,
    claims: JwtClaims = Depends(verify_jwt),
):
    """Stream raw document bytes (for inline preview). JWT-protected."""
    require_client_match(client, claims)
    try:
        path = storage.resolve_source(client, blueprint_code, source)
    except FileNotFoundError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"not found: {source}")
    media = "application/pdf" if source.lower().endswith(".pdf") else "application/octet-stream"
    return FileResponse(path, media_type=media, filename=source)


SYSTEM_PROMPT = (
    "Eres un asistente que responde preguntas sobre manuales de procesos "
    "productivos. Usa EXCLUSIVAMENTE la información de los fragmentos provistos. "
    "Si los fragmentos no contienen la respuesta, responde exactamente: "
    "'No encontré información relevante en los documentos.' "
    "Cita las fuentes al final con el formato [fuente, página N]."
)


def _build_prompt(question: str, hits: list[retrieve.Hit]) -> list[dict]:
    context_blocks = []
    for i, h in enumerate(hits, start=1):
        context_blocks.append(
            f"[fragmento {i} — fuente: {h.source}, página {h.page}]\n{h.text}")
    context = "\n\n".join(context_blocks) if context_blocks else "(sin fragmentos)"
    user = f"Contexto:\n{context}\n\nPregunta: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


@router.post("/chat")
async def chat(
    client: str,
    blueprint_code: str,
    question: str,
    top_k: int = 5,
    claims: JwtClaims = Depends(verify_jwt),
):
    """Stream a grounded answer as Server-Sent Events.

    Event types:
      - `token`   : data = {"text": "..."}       (incremental chunks)
      - `citations`: data = {"citations": [...]} (emitted once before the first token)
      - `done`    : data = {}
    """
    require_client_match(client, claims)

    # 1) Embed the question, 2) retrieve, 3) assemble prompt, 4) stream.
    q_vec = await embeddings.embed_one(question)
    hits = retrieve.search(client, blueprint_code, q_vec, top_k=top_k)
    hits = [h for h in hits if h.score >= MIN_RETRIEVAL_SCORE]

    async def event_stream() -> AsyncIterator[dict]:
        citations = [
            Citation(source=h.source, page=h.page, chunk_idx=h.chunk_idx, score=h.score).model_dump()
            for h in hits
        ]
        yield {"event": "citations", "data": json.dumps({"citations": citations})}

        if not hits:
            yield {"event": "token",
                   "data": json.dumps({"text": "No encontré información relevante en los documentos."})}
            yield {"event": "done", "data": "{}"}
            return

        messages = _build_prompt(question, hits)
        async for token in llm.chat_stream(messages):
            yield {"event": "token", "data": json.dumps({"text": token})}
        yield {"event": "done", "data": "{}"}

    return EventSourceResponse(event_stream())
