"""API HTTP del RAG.

    POST /documents   sube un archivo (PDF, txt, md, docx) y lo indexa
    GET  /documents   lista lo que hay indexado
    POST /ask         responde una pregunta con fuentes

Uso: uvicorn api.main:app --reload
La documentación interactiva queda en http://localhost:8000/docs
"""
import os
import re
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from llama_index.core import VectorStoreIndex
from pydantic import BaseModel, Field

import config
import rag

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Estado compartido del proceso: un único índice y su query engine.
# El lock evita que dos subidas simultáneas modifiquen el índice a la vez.
state: dict = {"index": None, "engine": None}
index_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Al arrancar: configura los modelos y carga el índice (o crea uno vacío)."""
    config.configure_models()
    if os.path.isdir(config.STORAGE_DIR):
        index = rag.load_index()
    else:
        index = VectorStoreIndex([])  # se persiste con el primer documento
    state["index"] = index
    state["engine"] = rag.build_query_engine(index)
    yield


app = FastAPI(
    title="RAG API",
    description="Preguntas sobre documentos propios, con fuentes.",
    lifespan=lifespan,
)


# --- Modelos de entrada/salida (Pydantic valida y documenta automáticamente) ---


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000, examples=["¿Qué es RAG?"])


class Source(BaseModel):
    file_name: str
    page: str | None = None
    score: float | None = None


class AskResponse(BaseModel):
    answer: str
    sources: list[Source]


class DocumentInfo(BaseModel):
    file_name: str
    pages: int
    chunks: int


# --- Helpers ---

PROVIDER_ERRORS = (httpx.HTTPError, ConnectionError, TimeoutError)


def _provider_error(exc: Exception) -> HTTPException:
    """Traduce fallas del proveedor de modelos (Ollama caído, timeout...) a HTTP."""
    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException, ConnectionError, TimeoutError)):
        return HTTPException(
            status_code=503,
            detail=f"El proveedor de modelos ({config.PROVIDER}) no responde: {exc!r}",
        )
    return HTTPException(status_code=502, detail=f"Error del proveedor de modelos: {exc!r}")


def _safe_filename(name: str | None) -> str:
    name = os.path.basename(name or "").strip()
    return re.sub(r"[^\w.\-]", "_", name)


# --- Endpoints ---


@app.get("/documents", response_model=list[DocumentInfo])
def get_documents():
    return rag.list_documents(state["index"])


@app.post("/documents", response_model=DocumentInfo, status_code=201)
def upload_document(file: UploadFile = File(...)):
    file_name = _safe_filename(file.filename)
    ext = os.path.splitext(file_name)[1].lower()
    if not file_name or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Formato no soportado. Permitidos: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    dest = os.path.join(config.DATA_DIR, file_name)
    if os.path.exists(dest):
        raise HTTPException(status_code=409, detail=f"'{file_name}' ya está indexado")

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera los 50 MB")

    os.makedirs(config.DATA_DIR, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(content)

    try:
        with index_lock:
            rag.add_file(state["index"], dest)
    except PROVIDER_ERRORS as exc:
        os.remove(dest)  # no dejar en data/ algo que no quedó indexado
        raise _provider_error(exc) from exc
    except Exception as exc:
        os.remove(dest)
        raise HTTPException(
            status_code=422, detail=f"No se pudo leer o indexar el archivo: {exc}"
        ) from exc

    return next(d for d in rag.list_documents(state["index"]) if d["file_name"] == file_name)


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    if not state["index"].docstore.docs:
        raise HTTPException(status_code=409, detail="No hay documentos indexados todavía")
    try:
        text, sources = rag.answer(state["engine"], body.question.strip())
    except PROVIDER_ERRORS as exc:
        raise _provider_error(exc) from exc
    return {"answer": text, "sources": sources}
