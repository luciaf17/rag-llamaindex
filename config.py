"""Configuración del modelo según PROVIDER en .env: openai, ollama, gemini o vertex."""
import os

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter

load_dotenv()

PROVIDER = os.getenv("PROVIDER", "openai").lower()
DATA_DIR = os.getenv("DATA_DIR", "data")

# PostgreSQL con pgvector. Formato: postgresql://usuario:clave@host:puerto/base
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://rag:rag@localhost:5432/rag")
# Una tabla por proveedor: los embeddings de modelos distintos no se pueden mezclar.
# LlamaIndex le agrega el prefijo "data_" (rag_ollama -> data_rag_ollama).
PG_TABLE = os.getenv("PG_TABLE") or f"rag_{PROVIDER}"
# Dimensión del embedding. Si queda vacía se detecta con una llamada al modelo.
EMBED_DIM = int(os.getenv("EMBED_DIM") or 0)

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "64"))
TOP_K = int(os.getenv("TOP_K", "4"))


def configure_models() -> None:
    """Define LLM, embeddings y chunking globales para LlamaIndex."""
    if PROVIDER == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        from llama_index.llms.ollama import Ollama

        Settings.llm = Ollama(
            model=os.getenv("OLLAMA_LLM", "llama3.1"),
            request_timeout=120.0,
            temperature=0.1,
            # Sin esto la librería usa el contexto máximo del modelo (128k en
            # llama3.1): la caché KV no entra en la GPU y las consultas dan timeout.
            context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "8192")),
        )
        Settings.embed_model = OllamaEmbedding(
            model_name=os.getenv("OLLAMA_EMBED", "nomic-embed-text")
        )
    elif PROVIDER == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding
        from llama_index.llms.openai import OpenAI

        Settings.llm = OpenAI(
            model=os.getenv("OPENAI_LLM", "gpt-4o-mini"), temperature=0.1
        )
        Settings.embed_model = OpenAIEmbedding(
            model=os.getenv("OPENAI_EMBED", "text-embedding-3-small")
        )
    elif PROVIDER in ("gemini", "vertex"):
        # Gemini vía Google AI Studio (API key) o vía Vertex AI en un proyecto de GCP.
        from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
        from llama_index.llms.google_genai import GoogleGenAI

        llm_kwargs = {"temperature": 0.1}
        embed_kwargs = {}
        if os.getenv("GEMINI_LLM"):
            llm_kwargs["model"] = os.getenv("GEMINI_LLM")
        if os.getenv("GEMINI_EMBED"):
            embed_kwargs["model_name"] = os.getenv("GEMINI_EMBED")

        if PROVIDER == "vertex":
            project = os.getenv("GCP_PROJECT")
            if not project:
                raise ValueError("Con PROVIDER=vertex hay que definir GCP_PROJECT en .env")
            vertex = {"project": project, "location": os.getenv("GCP_LOCATION", "us-central1")}
            llm_kwargs["vertexai_config"] = vertex
            embed_kwargs["vertexai_config"] = vertex
        else:
            api_key = os.getenv("GOOGLE_API_KEY")
            if not api_key:
                raise ValueError("Con PROVIDER=gemini hay que definir GOOGLE_API_KEY en .env")
            llm_kwargs["api_key"] = api_key
            embed_kwargs["api_key"] = api_key

        Settings.llm = GoogleGenAI(**llm_kwargs)
        Settings.embed_model = GoogleGenAIEmbedding(**embed_kwargs)
    else:
        raise ValueError(
            f"PROVIDER desconocido: {PROVIDER!r} (usar 'openai', 'ollama', 'gemini' o 'vertex')"
        )

    Settings.node_parser = SentenceSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
