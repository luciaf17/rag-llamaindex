"""Tests de los endpoints de la API con modelos falsos y PostgreSQL real."""
from pathlib import Path

import httpx
import sqlalchemy

import rag

VACACIONES = (
    "Política de vacaciones de ACME.\n"
    "Cada empleado tiene 21 días hábiles de vacaciones por año.\n"
    "Las vacaciones se piden a Recursos Humanos con 30 días de anticipación.\n"
)
HORARIOS = (
    "Horarios de la oficina de ACME.\n"
    "La oficina abre de lunes a viernes de 9 a 18 horas.\n"
)
PAPER = Path(__file__).parent.parent / "data" / "Attention_Is_All_You_Need.pdf"


def upload(client, name: str, content: bytes | str, content_type: str = "text/plain"):
    if isinstance(content, str):
        content = content.encode("utf-8")
    return client.post("/documents", files={"file": (name, content, content_type)})


def ask(client, question: str):
    return client.post("/ask", json={"question": question})


# --- GET /documents ---


def test_list_is_empty_without_documents(client):
    response = client.get("/documents")
    assert response.status_code == 200
    assert response.json() == []


def test_list_shows_uploaded_documents(client):
    upload(client, "vacaciones.txt", VACACIONES)
    upload(client, "horarios.txt", HORARIOS)

    response = client.get("/documents")
    assert response.status_code == 200
    assert [d["file_name"] for d in response.json()] == ["horarios.txt", "vacaciones.txt"]


# --- POST /documents ---


def test_upload_text_file(client):
    response = upload(client, "vacaciones.txt", VACACIONES)
    assert response.status_code == 201
    assert response.json() == {"file_name": "vacaciones.txt", "pages": 0, "chunks": 1}


def test_upload_pdf_counts_pages(client):
    response = upload(client, PAPER.name, PAPER.read_bytes(), "application/pdf")
    assert response.status_code == 201
    body = response.json()
    assert body["pages"] == 15
    assert body["chunks"] > 15


def test_upload_same_file_twice_is_conflict(client):
    upload(client, "vacaciones.txt", VACACIONES)
    response = upload(client, "vacaciones.txt", VACACIONES)
    assert response.status_code == 409


def test_upload_unsupported_format(client):
    response = upload(client, "datos.csv", "a,b\n1,2\n", "text/csv")
    assert response.status_code == 415


def test_upload_empty_file(client):
    response = upload(client, "vacio.txt", b"")
    assert response.status_code == 400


def test_upload_unreadable_pdf_is_rejected_and_not_indexed(client):
    response = upload(client, "roto.pdf", b"esto no es un pdf", "application/pdf")
    assert response.status_code == 422
    assert client.get("/documents").json() == []


def test_upload_without_file(client):
    response = client.post("/documents")
    assert response.status_code == 422


# --- POST /ask ---


def test_ask_without_documents_is_conflict(client):
    response = ask(client, "¿Cuántos días de vacaciones hay?")
    assert response.status_code == 409


def test_ask_answers_from_documents_with_sources(client):
    upload(client, "vacaciones.txt", VACACIONES)
    upload(client, "horarios.txt", HORARIOS)

    response = ask(client, "¿Cuántos días hábiles de vacaciones tiene cada empleado?")
    assert response.status_code == 200
    body = response.json()
    assert "21 días hábiles" in body["answer"]
    assert "vacaciones.txt" in [s["file_name"] for s in body["sources"]]
    assert all(s["score"] is not None for s in body["sources"])


def test_ask_outside_documents_says_it_does_not_know(client):
    """Ante una pregunta sin respuesta en los documentos, se niega y no cita fuentes."""
    upload(client, "vacaciones.txt", VACACIONES)
    upload(client, "horarios.txt", HORARIOS)

    response = ask(client, "¿Cuál es la capital de Australia?")
    assert response.status_code == 200
    assert response.json() == {"answer": rag.NO_ANSWER, "sources": []}


def test_ask_empty_question(client):
    response = ask(client, "")
    assert response.status_code == 422


# --- Errores de dependencias externas ---


def test_ask_when_model_provider_is_down(client, monkeypatch):
    upload(client, "vacaciones.txt", VACACIONES)

    def provider_down(*args, **kwargs):
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(rag, "answer", provider_down)
    response = ask(client, "¿Cuántos días de vacaciones hay?")
    assert response.status_code == 503


def test_database_down(client, monkeypatch):
    def database_down():
        raise sqlalchemy.exc.OperationalError("SELECT 1", {}, Exception("connection refused"))

    monkeypatch.setattr(rag, "list_documents", database_down)
    response = client.get("/documents")
    assert response.status_code == 503
