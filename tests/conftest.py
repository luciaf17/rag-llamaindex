"""Fixtures compartidas.

Los tests de la API corren contra PostgreSQL de verdad (el de docker compose),
en una tabla propia que se crea y se borra en cada test. Los modelos se
reemplazan por mocks de LlamaIndex: no hacen falta claves ni Ollama.
"""
import re
from typing import ClassVar

import pytest
import sqlalchemy
from fastapi.testclient import TestClient
from llama_index.core import Settings
from llama_index.core.embeddings import MockEmbedding
from llama_index.core.llms import CompletionResponse, MockLLM
from llama_index.core.node_parser import SentenceSplitter

import config as rag_config  # "config" es un nombre reservado en los hooks de pytest
import rag

TEST_TABLE = "rag_test"


# --- Opción --run-eval para la evaluación con modelos reales ---


def pytest_addoption(parser):
    parser.addoption(
        "--run-eval",
        action="store_true",
        help="corre la evaluación con el proveedor real configurado en .env",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-eval"):
        return
    skip = pytest.mark.skip(reason="evaluación con modelos reales: correr con --run-eval")
    for item in items:
        if "eval" in item.keywords:
            item.add_marker(skip)


# --- Modelos falsos ---


class GroundedMockLLM(MockLLM):
    """MockLLM que imita a un modelo que respeta el prompt del RAG.

    El MockLLM original devuelve el prompt tal cual, así que nunca se negaría a
    responder. Este lee el contexto y la pregunta del prompt y contesta con la
    oración del contexto que más palabras comparte con la pregunta. Si ninguna
    comparte al menos dos, responde la frase de negativa, como pide el prompt.
    """

    STOPWORDS: ClassVar[set[str]] = {"cuál", "cuáles", "cuánto", "cuántos", "cuántas", "sobre", "según", "para", "tiene"}

    def _words(self, text: str) -> set[str]:
        return {w for w in re.findall(r"\w{4,}", text.lower()) if w not in self.STOPWORDS}

    def complete(self, prompt: str, formatted: bool = False, **kwargs) -> CompletionResponse:
        parts = prompt.split("---------------------")
        context = parts[1] if len(parts) >= 3 else ""
        question = prompt.split("Pregunta:")[-1].split("Respuesta:")[0]
        keywords = self._words(question)

        sentences = [s.strip() for s in re.split(r"[.\n]", context) if s.strip()]
        best = max(sentences, key=lambda s: len(keywords & self._words(s)), default="")
        if len(keywords & self._words(best)) < 2:
            return CompletionResponse(text=rag.NO_ANSWER)
        return CompletionResponse(text=best + ".")


def configure_mock_models() -> None:
    Settings.llm = GroundedMockLLM()
    # Devuelve el mismo vector para cualquier texto: todos los chunks empatan.
    # Con pocos documentos y TOP_K=4 se recuperan todos, que es lo que queremos.
    Settings.embed_model = MockEmbedding(embed_dim=8)
    Settings.node_parser = SentenceSplitter(chunk_size=512, chunk_overlap=64)


# --- Base de datos y cliente HTTP ---


def _drop_test_table() -> None:
    with rag.get_engine().begin() as conn:
        conn.execute(sqlalchemy.text(f"DROP TABLE IF EXISTS data_{TEST_TABLE}"))


@pytest.fixture(scope="session")
def database():
    """Saltea los tests que la usan si no hay PostgreSQL en DATABASE_URL."""
    try:
        with rag.get_engine().connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
    except sqlalchemy.exc.OperationalError:
        pytest.skip("No hay PostgreSQL en DATABASE_URL. Levantalo con: docker compose up -d db")


@pytest.fixture
def client(database, monkeypatch):
    """Cliente HTTP de la API con modelos falsos y una tabla vacía."""
    monkeypatch.setattr(rag_config, "PG_TABLE", TEST_TABLE)
    monkeypatch.setattr(rag_config, "EMBED_DIM", 0)
    monkeypatch.setattr(rag_config, "configure_models", configure_mock_models)
    _drop_test_table()

    from api.main import app

    with TestClient(app) as test_client:  # el "with" dispara el lifespan
        yield test_client
    _drop_test_table()
