"""API HTTP del RAG.

    POST /documents   sube un archivo (PDF, txt, md, docx) y lo indexa
    GET  /documents   lista lo que hay indexado
    POST /ask         responde una pregunta con fuentes

Uso: uvicorn api.main:app --reload
La documentación interactiva queda en http://localhost:8000/docs
"""
import os
import re
import tempfile
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.exc import DBAPIError

import config
import rag

ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# Estado compartido del proceso: el índice (una vista sobre la tabla de pgvector)
# y su query engine. El lock evita que dos subidas simultáneas del mismo archivo
# pasen a la vez el chequeo de duplicados.
state: dict = {"index": None, "engine": None}
index_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Al arrancar: configura los modelos y conecta el índice a PostgreSQL.

    Si la base no responde, la app no arranca: es mejor fallar al inicio que
    aceptar tráfico que va a fallar en cada pedido.
    """
    config.configure_models()
    index = rag.load_index()
    state["index"] = index
    state["engine"] = rag.build_query_engine(index)
    yield


app = FastAPI(
    title="RAG API",
    description="Preguntas sobre documentos propios, con fuentes.",
    lifespan=lifespan,
)


@app.exception_handler(DBAPIError)
async def database_error(request: Request, exc: DBAPIError):
    """Cualquier falla de conexión o consulta a PostgreSQL se responde con 503."""
    return JSONResponse(
        status_code=503,
        content={"detail": f"La base de datos no está disponible: {exc.orig!r}"},
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


def _find_document(file_name: str) -> dict | None:
    return next((d for d in rag.list_documents() if d["file_name"] == file_name), None)


@app.get("/documents", response_model=list[DocumentInfo])
def get_documents():
    return rag.list_documents()


@app.post("/documents", response_model=DocumentInfo, status_code=201)
def upload_document(file: UploadFile = File(...)):
    file_name = _safe_filename(file.filename)
    ext = os.path.splitext(file_name)[1].lower()
    if not file_name or ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=f"Formato no soportado. Permitidos: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    content = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not content:
        raise HTTPException(status_code=400, detail="El archivo está vacío")
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="El archivo supera los 50 MB")

    with index_lock:
        if _find_document(file_name):
            raise HTTPException(status_code=409, detail=f"'{file_name}' ya está indexado")

        # El archivo solo hace falta mientras se lee: el texto y los embeddings
        # quedan en PostgreSQL. Se escribe en una carpeta temporal que se borra
        # sola, así la API no depende del disco (en Cloud Run es efímero).
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, file_name)
            with open(path, "wb") as f:
                f.write(content)
            try:
                rag.add_file(state["index"], path)
            except PROVIDER_ERRORS as exc:
                raise _provider_error(exc) from exc
            except DBAPIError:
                raise  # lo resuelve el handler de base de datos (503)
            except Exception as exc:
                raise HTTPException(
                    status_code=422, detail=f"No se pudo leer o indexar el archivo: {exc}"
                ) from exc

    return _find_document(file_name)


@app.post("/ask", response_model=AskResponse)
def ask(body: AskRequest):
    if not rag.list_documents():
        raise HTTPException(status_code=409, detail="No hay documentos indexados todavía")
    try:
        text, sources = rag.answer(state["engine"], body.question.strip())
    except PROVIDER_ERRORS as exc:
        raise _provider_error(exc) from exc
    return {"answer": text, "sources": sources}
