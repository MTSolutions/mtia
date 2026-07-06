"""End-to-end router tests with Ollama stubbed.

Uses FastAPI's TestClient to exercise the real routes (JWT dependency, Qdrant
upsert, SSE emission) without requiring Ollama. The embed/llm shims keep tests
deterministic and fast.
"""
from __future__ import annotations

import io
import json

import jwt
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from modules.rag import embeddings, ingest as ingest_mod, llm, retrieve, router as router_mod, storage
from modules.rag.router import router as rag_router


SECRET = "secret"


def _token(client="acme", login="tester", locale="es"):
    # Mirror the real api's token shape: sub is an int (user id), pyramid_jwt
    # extra claims included. mtia must accept this per RFC-lenient decode.
    return jwt.encode(
        {"sub": 1746, "roles": "admin", "ntype": 49,
         "client": client, "login": login, "locale": locale},
        SECRET, algorithm="HS512")


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("JWT_SECRET", SECRET)
    application = FastAPI()
    application.include_router(rag_router)
    return application


@pytest.fixture
def client_with_stubs(app, monkeypatch, ephemeral_client):
    """TestClient with Ollama calls stubbed."""
    def _vec(seed: int) -> list[float]:
        v = [0.0] * 1024
        v[seed % 1024] = 1.0
        return v

    async def fake_embed(texts, model=None):
        return [_vec(hash(t) & 0xFFFF) for t in texts]

    async def fake_embed_one(text, model=None):
        return _vec(hash(text) & 0xFFFF)

    async def fake_chat_stream(messages, model=None, options=None):
        # Yield a predictable multi-token sequence so the test can verify streaming.
        for piece in ["La ", "respuesta ", "es ", "cuatro."]:
            yield piece

    # Patch both the modules that define the functions AND the router's local refs.
    monkeypatch.setattr(embeddings, "embed", fake_embed)
    monkeypatch.setattr(embeddings, "embed_one", fake_embed_one)
    monkeypatch.setattr(llm, "chat_stream", fake_chat_stream)
    monkeypatch.setattr(ingest_mod.embeddings, "embed", fake_embed)

    # Lower the refusal threshold so our fake vectors yield scores that count as hits.
    monkeypatch.setattr(router_mod, "MIN_RETRIEVAL_SCORE", -1.0)

    c = TestClient(app)
    yield c, ephemeral_client
    retrieve.drop_collection(ephemeral_client)


def _make_pdf_bytes(pages_text: list[str]) -> bytes:
    from reportlab.pdfgen import canvas  # type: ignore
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    for text in pages_text:
        c.drawString(72, 800, text[:80])
        c.showPage()
    c.save()
    return buf.getvalue()


def test_unauthenticated_returns_401(client_with_stubs):
    c, _ = client_with_stubs
    r = c.get("/rag/documents", params={"client": "acme", "blueprint_code": "empaque"})
    assert r.status_code == 401


def test_cross_tenant_returns_403(client_with_stubs):
    c, client = client_with_stubs
    r = c.get("/rag/documents",
              params={"client": "someone_else", "blueprint_code": "empaque"},
              headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.status_code == 403


def test_ingest_upload_and_list(client_with_stubs):
    c, client = client_with_stubs
    pdf = _make_pdf_bytes(["Primera pagina.", "Segunda pagina de prueba."])
    r = c.post(
        "/rag/ingest",
        data={"client": client, "blueprint_code": "empaque"},
        files={"file": ("manual.pdf", pdf, "application/pdf")},
        headers={"Authorization": f"JWT {_token(client=client)}"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source"] == "manual.pdf"
    assert body["chunks"] >= 1

    r = c.get("/rag/documents",
              params={"client": client, "blueprint_code": "empaque"},
              headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["source"] == "manual.pdf"
    assert docs[0]["chunks"] == body["chunks"]


def test_document_content_streams_bytes(client_with_stubs):
    c, client = client_with_stubs
    pdf = _make_pdf_bytes(["Ver este documento."])
    c.post("/rag/ingest",
           data={"client": client, "blueprint_code": "empaque"},
           files={"file": ("mirame.pdf", pdf, "application/pdf")},
           headers={"Authorization": f"JWT {_token(client=client)}"})
    r = c.get("/rag/documents/content",
              params={"client": client, "blueprint_code": "empaque", "source": "mirame.pdf"},
              headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content.startswith(b"%PDF")


def test_document_content_404_when_missing(client_with_stubs):
    c, client = client_with_stubs
    r = c.get("/rag/documents/content",
              params={"client": client, "blueprint_code": "empaque", "source": "nope.pdf"},
              headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.status_code == 404


def test_document_content_cross_tenant_403(client_with_stubs):
    c, _ = client_with_stubs
    r = c.get("/rag/documents/content",
              params={"client": "other", "blueprint_code": "empaque", "source": "x.pdf"},
              headers={"Authorization": f"JWT {_token(client='mine')}"})
    assert r.status_code == 403


def test_delete_document_removes_chunks_and_file(client_with_stubs):
    c, client = client_with_stubs
    pdf = _make_pdf_bytes(["Doc a borrar."])
    c.post("/rag/ingest",
           data={"client": client, "blueprint_code": "empaque"},
           files={"file": ("borrame.pdf", pdf, "application/pdf")},
           headers={"Authorization": f"JWT {_token(client=client)}"})
    r = c.delete("/rag/documents",
                 params={"client": client, "blueprint_code": "empaque", "source": "borrame.pdf"},
                 headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.status_code == 200
    assert r.json()["deleted_chunks"] >= 1

    r = c.get("/rag/documents",
              params={"client": client, "blueprint_code": "empaque"},
              headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.json()["documents"] == []


def _parse_sse_events(text: str) -> list[tuple[str, dict]]:
    events = []
    event = None
    data_lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("event:"):
            event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:"):].strip())
        elif not line and event:
            payload = json.loads("\n".join(data_lines)) if data_lines else {}
            events.append((event, payload))
            event, data_lines = None, []
    return events


def test_list_documents_includes_shared(client_with_stubs):
    c, client = client_with_stubs
    pdf_step = _make_pdf_bytes(["Operación de la despaletizadora L1."])
    pdf_shared = _make_pdf_bytes(["Manual de prevención de riesgos eléctricos."])

    c.post("/rag/ingest",
           data={"client": client, "blueprint_code": "despaletizadora_l1"},
           files={"file": ("operacion.pdf", pdf_step, "application/pdf")},
           headers={"Authorization": f"JWT {_token(client=client)}"})
    c.post("/rag/ingest",
           data={"client": client, "blueprint_code": storage.SHARED_CODE},
           files={"file": ("seguridad.pdf", pdf_shared, "application/pdf")},
           headers={"Authorization": f"JWT {_token(client=client)}"})

    r = c.get("/rag/documents",
              params={"client": client, "blueprint_code": "despaletizadora_l1"},
              headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.status_code == 200
    docs = r.json()["documents"]
    by_source = {d["source"]: d for d in docs}
    assert set(by_source) == {"operacion.pdf", "seguridad.pdf"}
    assert by_source["operacion.pdf"]["blueprint_code"] == "despaletizadora_l1"
    assert by_source["seguridad.pdf"]["blueprint_code"] == storage.SHARED_CODE

    # Querying the shared bucket directly only returns shared docs
    # (no fan-in of itself — see router.list_documents guard).
    r_shared = c.get("/rag/documents",
                     params={"client": client, "blueprint_code": storage.SHARED_CODE},
                     headers={"Authorization": f"JWT {_token(client=client)}"})
    assert [d["source"] for d in r_shared.json()["documents"]] == ["seguridad.pdf"]


def test_chat_streams_citations_tokens_and_done(client_with_stubs):
    c, client = client_with_stubs
    pdf = _make_pdf_bytes(["Instrucciones para el empaque de cajas."])
    c.post("/rag/ingest",
           data={"client": client, "blueprint_code": "empaque"},
           files={"file": ("manual.pdf", pdf, "application/pdf")},
           headers={"Authorization": f"JWT {_token(client=client)}"})

    r = c.post("/rag/chat",
               params={"client": client, "blueprint_code": "empaque",
                       "question": "¿Cómo empaco las cajas?", "top_k": 3},
               headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.status_code == 200
    events = _parse_sse_events(r.text)
    event_names = [e[0] for e in events]
    assert event_names[0] == "citations"
    assert event_names[-1] == "done"
    tokens = [payload["text"] for name, payload in events if name == "token"]
    assert "".join(tokens) == "La respuesta es cuatro."
    citations = events[0][1]["citations"]
    assert any(c["source"] == "manual.pdf" for c in citations)


def test_chat_refuses_when_no_relevant_chunks(client_with_stubs, monkeypatch):
    c, client = client_with_stubs
    # Raise threshold so nothing passes.
    monkeypatch.setattr(router_mod, "MIN_RETRIEVAL_SCORE", 99.0)

    r = c.post("/rag/chat",
               params={"client": client, "blueprint_code": "empaque",
                       "question": "¿Cuál es la capital de Francia?"},
               headers={"Authorization": f"JWT {_token(client=client)}"})
    assert r.status_code == 200
    events = _parse_sse_events(r.text)
    tokens = [p["text"] for n, p in events if n == "token"]
    assert "No encontré información relevante" in "".join(tokens)
