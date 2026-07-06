"""Filesystem helpers for the RAG document tree.

Layout:
    $DOCUMENTS_DIR/<client>/<blueprint_code>/<source_filename>

Source filename acts as the document ID; re-uploading replaces it.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path


SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

# Reserved blueprint_code for documents that apply to every step of a client.
# Leading/trailing underscores are stripped by the frontend slugify(), so no
# user-authored step name can produce this value.
SHARED_CODE = "__shared__"


def documents_root() -> Path:
    return Path(os.environ.get("DOCUMENTS_DIR", "/app/documents"))


def _assert_safe(name: str, label: str) -> None:
    if not name or not SAFE_NAME.match(name):
        raise ValueError(f"unsafe {label}: {name!r}")


@dataclass
class DocumentInfo:
    source: str
    size_bytes: int
    mtime: float
    path: Path


def blueprint_dir(client: str, blueprint_code: str) -> Path:
    _assert_safe(client, "client")
    _assert_safe(blueprint_code, "blueprint_code")
    return documents_root() / client / blueprint_code


def list_documents(client: str, blueprint_code: str) -> list[DocumentInfo]:
    d = blueprint_dir(client, blueprint_code)
    if not d.is_dir():
        return []
    out: list[DocumentInfo] = []
    for entry in sorted(d.iterdir()):
        if entry.is_file() and not entry.name.startswith("."):
            stat = entry.stat()
            out.append(DocumentInfo(
                source=entry.name,
                size_bytes=stat.st_size,
                mtime=stat.st_mtime,
                path=entry,
            ))
    return out


def list_shared_documents(client: str) -> list[DocumentInfo]:
    return list_documents(client, SHARED_CODE)


def list_clients() -> list[str]:
    root = documents_root()
    if not root.is_dir():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def list_blueprints(client: str) -> list[str]:
    _assert_safe(client, "client")
    d = documents_root() / client
    if not d.is_dir():
        return []
    return sorted(p.name for p in d.iterdir() if p.is_dir())


def resolve_source(client: str, blueprint_code: str, source: str) -> Path:
    _assert_safe(source, "source")
    path = blueprint_dir(client, blueprint_code) / source
    if not path.is_file():
        raise FileNotFoundError(str(path))
    return path


def save_upload(client: str, blueprint_code: str, source: str, data: bytes) -> Path:
    _assert_safe(source, "source")
    d = blueprint_dir(client, blueprint_code)
    d.mkdir(parents=True, exist_ok=True)
    path = d / source
    path.write_bytes(data)
    return path


def delete_source(client: str, blueprint_code: str, source: str) -> bool:
    try:
        path = resolve_source(client, blueprint_code, source)
    except FileNotFoundError:
        return False
    path.unlink()
    return True
