"""Evaluación del RAG con el proveedor real (antes evaluate.py).

Corre cada pregunta de eval_set.json contra los documentos indexados y el
modelo configurado en .env:
  - "expected": palabras que la respuesta correcta debería contener
  - "out_of_scope": true si la respuesta NO está en los documentos
    (el sistema debería negarse en vez de inventar)

Es opcional porque es lenta y necesita modelos y documentos indexados:

    pytest --run-eval -m eval -v
"""
import json
from pathlib import Path

import pytest
import sqlalchemy

import config
import rag

pytestmark = pytest.mark.eval

CASES = json.loads((Path(__file__).parent.parent / "eval_set.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def query_engine():
    config.configure_models()
    try:
        index = rag.load_index()
        documents = rag.list_documents()
    except sqlalchemy.exc.OperationalError:
        pytest.skip("No hay PostgreSQL en DATABASE_URL. Levantalo con: docker compose up -d db")
    if not documents:
        pytest.skip(f"La tabla de {config.PROVIDER} está vacía. Corré primero: python rag.py ingest")
    return rag.build_query_engine(index)


@pytest.mark.parametrize("case", CASES, ids=[f"q{i}" for i in range(1, len(CASES) + 1)])
def test_eval_case(query_engine, case):
    answer, _ = rag.answer(query_engine, case["question"])
    refused = rag.NO_ANSWER.lower() in answer.lower()

    if case.get("out_of_scope"):
        assert refused, f"inventó una respuesta a '{case['question']}': {answer}"
    else:
        assert not refused, f"se negó sin motivo a '{case['question']}'"
        missing = [w for w in case.get("expected", []) if w.lower() not in answer.lower()]
        assert not missing, f"faltan {missing} en la respuesta: {answer}"
