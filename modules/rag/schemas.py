"""Pydantic request/response models for /rag/* endpoints."""
from __future__ import annotations

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    source: str
    size_bytes: int
    mtime: float
    chunks: int
    blueprint_code: str


class DocumentListResponse(BaseModel):
    client: str
    blueprint_code: str
    documents: list[DocumentInfo]


class ChatRequest(BaseModel):
    client: str
    blueprint_code: str
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class Citation(BaseModel):
    source: str
    page: int
    chunk_idx: int
    score: float


class IngestResponse(BaseModel):
    source: str
    chunks: int
    pages: int
    seconds: float


class DeleteResponse(BaseModel):
    source: str
    deleted_chunks: int
