"""RAG sobre documentos propios con LlamaIndex.

Uso:
    python rag.py ingest              # indexa los archivos de data/
    python rag.py ask "tu pregunta"   # una pregunta puntual
    python rag.py chat                # modo interactivo
"""
import os
import sys

from llama_index.core import (
    PromptTemplate,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)

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


def ingest() -> VectorStoreIndex:
    """Lee los documentos, los parte en chunks, genera embeddings y persiste el índice."""
    if not os.path.isdir(config.DATA_DIR) or not os.listdir(config.DATA_DIR):
        sys.exit(f"No hay documentos en '{config.DATA_DIR}/'. Copiá ahí tus PDFs o .txt.")

    documents = SimpleDirectoryReader(config.DATA_DIR, recursive=True).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    index.storage_context.persist(persist_dir=config.STORAGE_DIR)

    n_chunks = len(index.docstore.docs)
    print(f"\nIndexados {len(documents)} documentos/páginas en {n_chunks} chunks.")
    return index


def load_index() -> VectorStoreIndex:
    if not os.path.isdir(config.STORAGE_DIR):
        sys.exit("Todavía no hay índice. Corré primero: python rag.py ingest")
    storage = StorageContext.from_defaults(persist_dir=config.STORAGE_DIR)
    return load_index_from_storage(storage)


def build_query_engine(index: VectorStoreIndex):
    return index.as_query_engine(
        similarity_top_k=config.TOP_K, text_qa_template=QA_PROMPT
    )


def ask(query_engine, question: str, show_sources: bool = True) -> str:
    response = query_engine.query(question)
    answer = str(response).strip()
    print(f"\n{answer}")

    if show_sources and response.source_nodes and NO_ANSWER not in answer:
        print("\nFuentes:")
        for node in response.source_nodes:
            meta = node.node.metadata
            page = f" (pág. {meta['page_label']})" if meta.get("page_label") else ""
            score = f"{node.score:.2f}" if node.score is not None else "-"
            print(f"  · {meta.get('file_name', '?')}{page} — similitud {score}")
    return answer


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

    query_engine = build_query_engine(load_index())
    if command == "ask":
        if len(sys.argv) < 3:
            sys.exit('Falta la pregunta: python rag.py ask "..."')
        ask(query_engine, " ".join(sys.argv[2:]))
    else:
        chat(query_engine)


if __name__ == "__main__":
    main()
