"""RAG sobre documentos propios con LlamaIndex.

Uso:
    python rag.py ingest              # indexa los archivos de data/
    python rag.py ask "tu pregunta"   # una pregunta puntual
    python rag.py chat                # modo interactivo

Los chunks y sus embeddings se guardan en PostgreSQL con pgvector (DATABASE_URL).
"""
import os
import sys
from collections import defaultdict

import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine
from llama_index.core import PromptTemplate, Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.vector_stores.postgres import PGVectorStore

import config

NO_ANSWER = "No encuentro esa información en los documentos."

QA_PROMPT = PromptTemplate(
    "Sos un asistente que responde SOLO con la información del contexto.\n"
    "Si el contexto no alcanza para responder, contestá exactamente: "
    f'"{NO_ANSWER}"\n'
    "No inventes datos. Respondé en español, de forma breve y concreta.\n\n"
    "Contexto:\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Pregunta: {query_str}\n"
    "Respuesta: "
)


# --- Conexión a PostgreSQL ---

_engine: sqlalchemy.engine.Engine | None = None


def _db_url() -> sqlalchemy.engine.URL:
    url = config.DATABASE_URL
    if url.startswith("postgres://"):  # formato viejo que SQLAlchemy 2 no acepta
        url = "postgresql://" + url[len("postgres://"):]
    return sqlalchemy.make_url(url)


def get_engine() -> sqlalchemy.engine.Engine:
    """Un único pool de conexiones por proceso, compartido con el vector store."""
    global _engine
    if _engine is None:
        # pool_pre_ping descarta conexiones que el servidor cerró por inactividad.
        _engine = sqlalchemy.create_engine(_db_url(), pool_pre_ping=True)
    return _engine


def _table() -> str:
    # Mismo nombre que arma PGVectorStore: prefijo data_ y en minúsculas.
    return f"data_{config.PG_TABLE}".lower()


def _table_dim() -> int | None:
    """Dimensión de la columna embedding si la tabla ya existe, o None."""
    with get_engine().connect() as conn:
        return conn.execute(
            sqlalchemy.text(
                "SELECT atttypmod FROM pg_attribute "
                "WHERE attrelid = to_regclass(:t) AND attname = 'embedding'"
            ),
            {"t": _table()},
        ).scalar()


def build_vector_store() -> PGVectorStore:
    dim = config.EMBED_DIM or len(Settings.embed_model.get_text_embedding("dimensión"))
    existing = _table_dim()
    if existing and existing != dim:
        raise RuntimeError(
            f"La tabla '{_table()}' tiene embeddings de {existing} dimensiones y el modelo "
            f"actual produce {dim}. Usá otro PG_TABLE o borrá la tabla y reindexá."
        )
    # La librería exige también un engine asíncrono. Solo usamos los métodos
    # sincrónicos, y crear el engine no abre conexiones, así que no se usa.
    async_engine = create_async_engine(
        _db_url().set(drivername="postgresql+asyncpg", query={})
    )
    return PGVectorStore(
        engine=get_engine(),
        async_engine=async_engine,
        table_name=config.PG_TABLE,
        embed_dim=dim,
        use_jsonb=True,
        indexed_metadata_keys={("file_name", "text")},
        initialization_fail_on_error=True,
    )


# --- Indexación ---


def _index_documents(index: VectorStoreIndex, documents: list, show_progress: bool = False) -> int:
    """Indexa documentos reemplazando lo que ya hubiera de esos mismos archivos.

    Por cada archivo: guarda los ids de sus chunks viejos, inserta los nuevos y
    recién entonces borra los viejos. Así, si algo falla a mitad de camino, el
    archivo queda con su versión anterior en vez de desaparecer o duplicarse.
    Devuelve la cantidad de chunks insertados.
    """
    by_file: dict[str, list] = defaultdict(list)
    for doc in documents:
        # LlamaIndex mete la ruta completa en el texto que se embebe y que ve el
        # LLM. Se reemplaza por el nombre: si no, el mismo archivo da chunks
        # distintos según desde dónde se cargue, y la ruta local queda en la base.
        doc.metadata["file_path"] = doc.metadata.get("file_name", "?")
        # El nombre lo ve el LLM (sirve para citar), pero no entra al embedding:
        # si no, una pregunta que menciona el título de un paper empata con todos
        # sus chunks por igual y el que tiene la respuesta queda afuera del top-k.
        if "file_path" not in doc.excluded_embed_metadata_keys:
            doc.excluded_embed_metadata_keys.append("file_path")
        by_file[doc.metadata.get("file_name", "?")].append(doc)

    vector_store = index.vector_store
    total = 0
    for file_name, docs in by_file.items():
        same_file = MetadataFilters(filters=[MetadataFilter(key="file_name", value=file_name)])
        old_ids = [n.node_id for n in vector_store.get_nodes(filters=same_file)]
        nodes = Settings.node_parser.get_nodes_from_documents(docs)
        index.insert_nodes(nodes, show_progress=show_progress)
        if old_ids:
            vector_store.delete_nodes(node_ids=old_ids)
        total += len(nodes)
    return total


def ingest() -> VectorStoreIndex:
    """Lee los documentos de data/, los parte en chunks, genera embeddings y los guarda.

    Reindexar un archivo reemplaza sus chunks; los documentos subidos por la API
    que no están en data/ no se tocan.
    """
    if not os.path.isdir(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        sys.exit(f"No hay documentos en '{config.DATA_DIR}/'. Copiá ahí tus PDFs o .txt.")

    documents = SimpleDirectoryReader(config.DATA_DIR, recursive=True).load_data()
    documents = [d for d in documents if d.text.strip()]
    index = load_index()
    n_chunks = _index_documents(index, documents, show_progress=True)
    print(f"\nIndexados {len(documents)} documentos/páginas en {n_chunks} chunks.")
    return index


def load_index() -> VectorStoreIndex:
    """Índice respaldado por pgvector. No carga nada en memoria: consulta la base."""
    return VectorStoreIndex.from_vector_store(build_vector_store())


def build_query_engine(index: VectorStoreIndex):
    return index.as_query_engine(
        similarity_top_k=config.TOP_K, text_qa_template=QA_PROMPT
    )


def answer(query_engine, question: str) -> tuple[str, list[dict]]:
    """Responde una pregunta y devuelve (texto, fuentes) como datos, sin imprimir.

    Cada fuente es {"file_name", "page", "score"}. Si el modelo se negó a
    responder, la lista de fuentes va vacía. Lo usan la CLI y la API.
    """
    response = query_engine.query(question)
    text = str(response).strip()
    sources: list[dict] = []
    if NO_ANSWER not in text:
        for node in response.source_nodes:
            meta = node.node.metadata
            sources.append(
                {
                    "file_name": meta.get("file_name", "?"),
                    "page": meta.get("page_label"),
                    "score": round(node.score, 3) if node.score is not None else None,
                }
            )
    return text, sources


def ask(query_engine, question: str, show_sources: bool = True) -> str:
    text, sources = answer(query_engine, question)
    print(f"\n{text}")

    if show_sources and sources:
        print("\nFuentes:")
        for src in sources:
            page = f" (pág. {src['page']})" if src["page"] else ""
            score = f"{src['score']:.2f}" if src["score"] is not None else "-"
            print(f"  · {src['file_name']}{page} — similitud {score}")
    return text


def add_file(index: VectorStoreIndex, path: str) -> int:
    """Indexa un archivo dentro del índice existente, sin reconstruir todo.

    Parte el archivo en chunks, calcula sus embeddings y los guarda en la base.
    Devuelve la cantidad de chunks insertados.
    """
    documents = SimpleDirectoryReader(input_files=[path]).load_data()
    # El lector no lanza error ante un archivo ilegible: lo saltea y devuelve
    # vacío. Tampoco hay texto en un PDF escaneado. En ambos casos, fallar.
    documents = [d for d in documents if d.text.strip()]
    if not documents:
        raise ValueError("no se pudo extraer texto (¿archivo dañado o PDF escaneado?)")
    return _index_documents(index, documents)


def list_documents() -> list[dict]:
    """Archivos indexados con cantidad de páginas y chunks, calculado en la base."""
    with get_engine().connect() as conn:
        if conn.execute(sqlalchemy.text("SELECT to_regclass(:t)"), {"t": _table()}).scalar() is None:
            return []  # la tabla se crea con el primer documento
        rows = conn.execute(
            sqlalchemy.text(
                f"SELECT metadata_->>'file_name' AS file_name, "
                f"count(DISTINCT metadata_->>'page_label') AS pages, count(*) AS chunks "
                f"FROM {_table()} GROUP BY 1 ORDER BY 1"
            )
        )
        return [dict(row._mapping) for row in rows]


def chat(query_engine) -> None:
    print("Modo chat. Escribí 'salir' para terminar.")
    while True:
        try:
            question = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if question.lower() in {"salir", "exit", "quit"}:
            break
        if question:
            ask(query_engine, question)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in {"ingest", "ask", "chat"}:
        sys.exit(__doc__)

    config.configure_models()
    command = sys.argv[1]

    if command == "ingest":
        ingest()
        return

    index = load_index()
    if not list_documents():
        sys.exit("Todavía no hay documentos indexados. Corré primero: python rag.py ingest")
    query_engine = build_query_engine(index)
    if command == "ask":
        if len(sys.argv) < 3:
            sys.exit('Falta la pregunta: python rag.py ask "..."')
        ask(query_engine, " ".join(sys.argv[2:]))
    else:
        chat(query_engine)


if __name__ == "__main__":
    main()
