# Imagen de la API del RAG. Sirve igual para docker compose y para Cloud Run.
FROM python:3.11-slim

# Sin .pyc en disco y logs sin buffer (se ven en tiempo real en docker logs).
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Primero solo las dependencias: esta capa queda en caché y no se reinstala
# todo cada vez que cambia una línea de código.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY config.py rag.py ./
COPY api/ api/

# No correr como root dentro del contenedor.
RUN useradd --create-home --uid 1000 app
USER app

# Cloud Run indica el puerto en la variable PORT; en local se usa 8000.
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
