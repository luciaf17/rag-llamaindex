# RAG sobre documentos propios

Asistente que responde preguntas usando **solo** el contenido de tus documentos (PDF, .txt, .md, .docx) y cita de qué archivo y página sale cada respuesta. Si la información no está en los documentos, lo dice en vez de inventar.

Se usa desde una **interfaz web**, una **API REST** o la **línea de comandos**. Los fragmentos y sus embeddings se guardan en **PostgreSQL con pgvector**. Los modelos pueden ser locales con **Ollama** (gratis y sin que los datos salgan de tu máquina), **Gemini** sobre Vertex AI o con API key, u **OpenAI**. Se cambia con una variable.

![Interfaz web: una pregunta respondida con sus fuentes y otra fuera de los documentos](docs/interfaz.png)

## Arquitectura

```
  Navegador (HTMX + Tailwind)     Clientes HTTP (JSON)      CLI
            │ fragmentos HTML            │                    │
            ▼                            ▼                    │
      ┌──────────────── FastAPI (api/) ───────────────┐       │
      │  GET /  ·  /ui/*            /documents · /ask │       │
      └───────────────────────┬───────────────────────┘       │
                              ▼                               ▼
                    ┌────────────────── rag.py (LlamaIndex) ────────────────┐
                    │  ingesta · chunking · recuperación · prompt · fuentes │
                    └───────────┬───────────────────────────┬───────────────┘
                                ▼                           ▼
                  Proveedor de modelos             PostgreSQL + pgvector
           (Ollama · Vertex AI · Gemini · OpenAI)  (texto, metadatos y embeddings)
```

**Ingesta.** Se leen los documentos y se parten en fragmentos de 512 tokens con 64 de solapamiento, para no cortar ideas a la mitad. Cada fragmento se convierte en un embedding y se guarda en PostgreSQL. Reindexar un archivo reemplaza sus fragmentos anteriores, así que el ingest se puede correr varias veces sin duplicar.

**Consulta.** La pregunta se convierte en embedding y pgvector devuelve los 4 fragmentos más parecidos por similitud coseno. El modelo los recibe con un prompt que lo obliga a responder solo con ese contexto, o a contestar exactamente "No encuentro esa información en los documentos." Cuando se niega, no se muestran fuentes.

**Dónde corre cada pieza:**

| Entorno | API | Base de datos | Modelos |
|---|---|---|---|
| Local con Docker | Contenedor `api` | Contenedor `db` (pgvector) | Ollama en tu máquina, u otro proveedor |
| Producción | Google Cloud Run | Supabase | Gemini sobre Vertex AI |

**Estructura del repo:**

```
api/main.py          endpoints JSON y rutas de la interfaz
api/ui.py            fragmentos HTML que devuelve el servidor
api/static/          la página (HTMX y Tailwind por CDN, sin build)
rag.py               lógica del RAG y CLI
config.py            proveedor de modelos y parámetros, desde variables de entorno
tests/               pytest: API, interfaz y evaluación opcional
eval_set.json        preguntas de evaluación
data/                documentos de ejemplo (4 papers de arXiv)
Dockerfile           imagen de la API (local y Cloud Run)
docker-compose.yml   API + PostgreSQL con pgvector
```

## Documentos de ejemplo

`data/` trae cuatro papers de arXiv:

| Archivo | Paper | Páginas | Chunks |
|---|---|---|---|
| `Attention_Is_All_You_Need.pdf` | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | 15 | 29 |
| `Retrieval-Augmented_Generation_for_Knowledge-Intensive_NLP_Tasks.pdf` | [RAG for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020) | 19 | 52 |
| `ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.pdf` | [ReAct](https://arxiv.org/abs/2210.03629) (Yao et al., 2022) | 33 | 91 |
| `Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf` | [Lost in the Middle](https://arxiv.org/abs/2307.03172) (Liu et al., 2023) | 18 | 51 |

Ejemplo real con Ollama (llama3.1 + nomic-embed-text):

```
$ python rag.py ask "¿Qué pasa con el rendimiento cuando la información relevante está en el medio del contexto?"

El rendimiento de los modelos de lenguaje decae significativamente cuando deben acceder y
utilizar información ubicada en el medio de su contexto.

Fuentes:
  · Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf (pág. 1) — similitud 0.53
  · Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf (pág. 5) — similitud 0.52
  · Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf (pág. 8) — similitud 0.50
  · Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf (pág. 16) — similitud 0.50
```

## Inicio rápido con Docker

Requisitos: Docker y un proveedor de modelos. Con Ollama (instalado en tu máquina, fuera de Docker):

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Después:

```bash
cp .env.example .env                                # cambiá PROVIDER a ollama, o completá la clave de otro
docker compose up -d --build                        # levanta la API y PostgreSQL
docker compose exec api python rag.py ingest        # indexa los PDFs de data/
```

- Interfaz: http://localhost:8000
- Documentación interactiva de la API: http://localhost:8000/docs

Cómo está armado el compose:

- **`db`** usa la imagen `pgvector/pgvector:pg17` y guarda los datos en el volumen `rag-pgdata`, que sobrevive a `docker compose down`. La API espera a que su healthcheck dé OK antes de arrancar.
- **`api`** se construye con el `Dockerfile`, lee tu `.env` y pisa dos valores: la base pasa a ser `db` y Ollama pasa a ser `host.docker.internal`, porque dentro de un contenedor `localhost` es el propio contenedor.
- **`data/`** se monta de solo lectura: agregás PDFs y corrés el ingest sin reconstruir la imagen.
- **Ollama tiene que estar corriendo antes del `up`.** Al arrancar, la API pide un embedding para detectar su dimensión; si Ollama no responde, Docker la reintenta hasta que responda.

```bash
docker compose logs -f api    # ver logs
docker compose down           # apagar (los datos quedan en el volumen)
```

## Desarrollo sin Docker

```bash
python -m venv .venv && source .venv/bin/activate    # en Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env
docker compose up -d db                               # solo la base

python rag.py ingest                                  # indexa data/
python rag.py ask "¿Qué es la atención multi-cabeza?"
python rag.py chat                                    # modo interactivo
uvicorn api.main:app --reload                         # API e interfaz en http://localhost:8000
```

En Windows, definí `PYTHONUTF8=1` (`$env:PYTHONUTF8=1` en PowerShell). La consola usa cp1252 por defecto y no puede imprimir algunos caracteres de las respuestas.

## API

| Método y ruta | Qué hace | Respuestas |
|---|---|---|
| `GET /documents` | Lista los archivos indexados con páginas y chunks | 200 |
| `POST /documents` | Sube un archivo (campo `file`) y lo indexa | 201 · 400 vacío · 409 ya indexado · 413 más de 50 MB · 415 formato no soportado · 422 sin texto legible |
| `POST /ask` | Responde `{"question": "..."}` con respuesta y fuentes | 200 · 409 sin documentos · 422 pregunta vacía |

Si la base no responde, cualquier endpoint devuelve 503. Si no responde el proveedor de modelos, subir y preguntar también devuelven 503.

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "¿Qué pasa con el rendimiento cuando la información relevante está en el medio del contexto?"}'
```

```json
{
  "answer": "El rendimiento de los modelos de lenguaje decae significativamente cuando deben acceder y utilizar información ubicada en el medio de su contexto.",
  "sources": [
    {"file_name": "Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf", "page": "1", "score": 0.526},
    {"file_name": "Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf", "page": "5", "score": 0.522},
    {"file_name": "Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf", "page": "8", "score": 0.504},
    {"file_name": "Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf", "page": "16", "score": 0.501}
  ]
}
```

Los archivos subidos no se guardan en disco: se leen desde una carpeta temporal, se indexan y se borran. La base es la única fuente de verdad, lo que permite correr la API en Cloud Run, donde el disco es efímero.

## Interfaz web

Una sola página en `api/static/index.html`, con HTMX y Tailwind cargados por CDN (versiones fijas y hash de integridad). No hay React ni paso de build.

- **El servidor devuelve HTML, no JSON.** HTMX envía el formulario y reemplaza una parte de la página con el fragmento que llega. Los fragmentos se arman en `api/ui.py`, y todo texto externo se escapa para evitar inyección de scripts (XSS).
- **La pregunta aparece al instante.** `/ui/ask` devuelve la pregunta y un recuadro "Pensando…" con `hx-trigger="load"`, que pide la respuesta a `/ui/answer` y se reemplaza con ella. Si tarda más de 8 segundos, se avisa que el modelo puede estar cargándose.
- **La lista de documentos se actualiza sola.** Después de una subida, el servidor envía el encabezado `HX-Trigger: documents-changed` y la lista se recarga.
- **Los errores se ven.** Las rutas `/ui/*` reusan los endpoints JSON y devuelven sus errores como HTML con el mismo código de estado.

## Tests

```bash
docker compose up -d db
pytest                          # 24 tests con modelos falsos, unos 6 segundos
pytest --run-eval -m eval -v    # evaluación con el proveedor real de .env
```

- **Los tests de la API y la interfaz** usan `MockEmbedding` y un `MockLLM` que responde con la oración del contexto que más palabras comparte con la pregunta, o se niega si ninguna alcanza. No necesitan claves ni Ollama.
- **Corren contra PostgreSQL real**, en la tabla `data_rag_test`, que se crea y se borra en cada test. Si no hay base, se saltean con un mensaje que dice cómo levantarla.
- **Cubren** los tres endpoints con sus errores, la negativa ante preguntas fuera de los documentos, los 503 cuando falla un servicio externo y el escape de HTML.
- **Qué no miden:** la calidad de la búsqueda. `MockEmbedding` devuelve el mismo vector para todo texto. De eso se encarga la evaluación.

## Evaluación

`eval_set.json` tiene 6 preguntas sobre los papers. Cuatro tienen respuesta en los documentos y se verifica que contenga palabras clave. Dos están fuera de alcance, sobre la capital de Australia y el Mundial 2022, y se verifica que el sistema se niegue en vez de inventar. Cada pregunta es un test de `tests/test_eval.py`.

Resultados con `PROVIDER=ollama` (llama3.1 + nomic-embed-text), `CHUNK_SIZE=512`, `CHUNK_OVERLAP=64`, `TOP_K=4`, en una RTX 3070 de 8 GB:

| # | Pregunta | Esperado | Resultado |
|---|---|---|---|
| 1 | ¿Cuántas cabezas de atención usa el Transformer base? | `8` | ✓ |
| 2 | ¿Cuáles son las dos variantes de RAG y en qué se diferencian? | `Sequence`, `Token` | ✓ |
| 3 | ¿En qué dos benchmarks de razonamiento se evalúa ReAct? | `HotpotQA`, `FEVER` | ✓ |
| 4 | ¿Qué forma tiene la curva de rendimiento según la posición de la información? | `U` | ✓ |
| 5 | ¿Cuál es la capital de Australia? | se niega | ✓ |
| 6 | ¿Qué equipo ganó la Copa del Mundo 2022? | se niega | ✓ |

**6/6 en dos corridas**, en unos 10 segundos con el modelo ya cargado.

Hallazgos al ajustar la configuración:

- **Qué se embebe importa.** LlamaIndex agrega la ruta del archivo al texto de cada fragmento. Con el nombre del archivo dentro del embedding, una pregunta que menciona el título de un paper empataba con todos sus fragmentos por igual, y el que tenía la respuesta quedaba afuera: 4/6. Excluyendo el nombre del embedding, dejándolo visible solo para el modelo, se volvió a 6/6.
- **Más contexto no es siempre mejor.** Con `TOP_K` de 5 o 6, la evaluación bajó a 5/6: con más fragmentos, llama3.1 de 8B se confunde en la pregunta sobre las variantes de RAG. Es el efecto que describe "Lost in the Middle".

## Despliegue en Google Cloud Run

La API corre en Cloud Run, la base en Supabase y los modelos son Gemini sobre Vertex AI. Esta guía no se probó de punta a punta: requiere cuentas de Google Cloud y Supabase.

**Decisiones:**

- **Vertex AI como proveedor.** Cloud Run no puede usar Ollama. Con Vertex, el servicio se autentica con su cuenta de servicio y no hay claves de API que guardar.
- **Supabase por el pooler en modo transacción** (puerto 6543). La conexión directa de Supabase es solo IPv6, y Cloud Run sale por IPv4 por defecto. El pooler además reparte pocas conexiones reales entre todas las instancias.
- **Servicio privado.** La API no tiene autenticación. Pública, cualquiera podría subir documentos y consumir tu cuota de Gemini. Se accede por un túnel autenticado con `gcloud run services proxy`.
- **Una sola instancia** (`--max-instances 1`). El candado que evita subidas duplicadas simultáneas vive dentro del proceso. También acota el gasto.

**Variables del servicio:**

| Variable | Valor | Cómo se pasa |
|---|---|---|
| `DATABASE_URL` | URI del Transaction pooler de Supabase, con `?sslmode=require` al final | Secret Manager |
| `PROVIDER` | `vertex` | variable de entorno |
| `GCP_PROJECT` | ID del proyecto de Google Cloud | variable de entorno |
| `GCP_LOCATION` | `us-central1` | variable de entorno |
| `GEMINI_LLM` | modelo de Gemini para responder | variable de entorno |
| `GEMINI_EMBED` | modelo de embeddings | variable de entorno |
| `EMBED_DIM` | dimensión del modelo de embeddings | variable de entorno |

Fijá `GEMINI_LLM` y `GEMINI_EMBED` con modelos estables del Model Garden de Vertex AI. Si quedan vacíos se usan los defaults de la librería, que cambian entre versiones, y cambiar el modelo de embeddings vuelve incomparables los vectores guardados. Fijar `EMBED_DIM` evita una llamada a Gemini en cada arranque en frío. `PORT` lo define Cloud Run.

**Pasos.** Se pueden correr en Cloud Shell, la terminal del navegador de Google Cloud, que ya trae `gcloud` y Python.

1. **Supabase.** Creá un proyecto y guardá la contraseña de la base. En Database → Extensions, activá `vector`. En **Connect**, copiá la URI de **Transaction pooler**, reemplazá la contraseña y agregá `?sslmode=require`.

2. **Proyecto y APIs.**

   ```bash
   gcloud config set project TU_PROYECTO
   export PROJECT_ID=$(gcloud config get-value project)
   export REGION=southamerica-east1
   export IMAGE=$REGION-docker.pkg.dev/$PROJECT_ID/rag/rag-api:v1
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
     artifactregistry.googleapis.com secretmanager.googleapis.com aiplatform.googleapis.com
   ```

3. **Construir la imagen.** Cloud Build usa el `Dockerfile` y guarda la imagen en Artifact Registry. `.gcloudignore` evita subir `.env`, el entorno virtual y los documentos.

   ```bash
   git clone https://github.com/luciaf17/rag-llamaindex.git && cd rag-llamaindex
   gcloud artifacts repositories create rag --repository-format=docker --location=$REGION
   gcloud builds submit --tag $IMAGE
   ```

4. **Cuenta de servicio** con permiso solo para usar Vertex AI.

   ```bash
   gcloud iam service-accounts create rag-api
   export SA=rag-api@$PROJECT_ID.iam.gserviceaccount.com
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member=serviceAccount:$SA --role=roles/aiplatform.user
   ```

5. **La URL de la base como secreto**, legible solo por esa cuenta.

   ```bash
   read -s DATABASE_URL        # pegá la URI de Supabase; no queda en el historial
   printf '%s' "$DATABASE_URL" | gcloud secrets create rag-database-url --data-file=-
   gcloud secrets add-iam-policy-binding rag-database-url \
     --member=serviceAccount:$SA --role=roles/secretmanager.secretAccessor
   ```

6. **Indexar los documentos en Supabase** y averiguar la dimensión del embedding. El último comando la imprime.

   ```bash
   python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
   export PROVIDER=vertex GCP_PROJECT=$PROJECT_ID GCP_LOCATION=us-central1
   export GEMINI_LLM=MODELO GEMINI_EMBED=MODELO_EMBED DATABASE_URL
   python rag.py ingest
   python -c "import config; config.configure_models(); from llama_index.core import Settings; print(len(Settings.embed_model.get_text_embedding('x')))"
   ```

7. **Desplegar.**

   ```bash
   gcloud run deploy rag-api --image $IMAGE --region $REGION \
     --service-account $SA \
     --set-env-vars PROVIDER=vertex,GCP_PROJECT=$PROJECT_ID,GCP_LOCATION=us-central1,GEMINI_LLM=MODELO,GEMINI_EMBED=MODELO_EMBED,EMBED_DIM=DIMENSION \
     --set-secrets DATABASE_URL=rag-database-url:latest \
     --memory 1Gi --cpu 1 --min-instances 0 --max-instances 1 \
     --no-allow-unauthenticated
   ```

8. **Abrir la interfaz.** El túnel expone el servicio privado en el puerto 8080. En Cloud Shell, se abre con **Web preview** en ese puerto.

   ```bash
   gcloud run services proxy rag-api --region $REGION --port 8080
   ```

9. **Actualizar o borrar.** Para publicar cambios, repetí el paso 3 con otra etiqueta y corré `gcloud run deploy rag-api --image <imagen nueva> --region $REGION`; el resto de la configuración se conserva. Para dar de baja el servicio, `gcloud run services delete rag-api --region $REGION`.

Costos a tener en cuenta: Vertex AI cobra por tokens, Artifact Registry por almacenamiento, y Cloud Run con `--min-instances 0` no cobra mientras no recibe pedidos. En el plan gratuito, Supabase pausa los proyectos inactivos.

## Configuración

Todas las variables se leen de `.env` o del entorno (ver `.env.example`).

| Variable | Default | Qué controla |
|---|---|---|
| `PROVIDER` | `openai` | `ollama`, `vertex`, `gemini` u `openai` |
| `DATABASE_URL` | `postgresql://rag:rag@localhost:5432/rag` | Conexión a PostgreSQL con pgvector |
| `PG_TABLE` | `rag_<PROVIDER>` | Tabla de fragmentos; LlamaIndex le agrega el prefijo `data_` |
| `EMBED_DIM` | se detecta | Dimensión del embedding |
| `CHUNK_SIZE` | `512` | Tamaño de cada fragmento, en tokens |
| `CHUNK_OVERLAP` | `64` | Solapamiento entre fragmentos |
| `TOP_K` | `4` | Fragmentos recuperados por pregunta |
| `DATA_DIR` | `data` | Carpeta que lee `rag.py ingest` |
| `OLLAMA_LLM` / `OLLAMA_EMBED` | `llama3.1` / `nomic-embed-text` | Modelos de Ollama |
| `OLLAMA_CONTEXT_WINDOW` | `8192` | Contexto del LLM en Ollama |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Dirección de Ollama; en compose, `host.docker.internal` |
| `OPENAI_API_KEY`, `OPENAI_LLM`, `OPENAI_EMBED` | `gpt-4o-mini`, `text-embedding-3-small` | OpenAI |
| `GOOGLE_API_KEY` | | Gemini con API key de Google AI Studio |
| `GCP_PROJECT`, `GCP_LOCATION` | `us-central1` | Gemini sobre Vertex AI |
| `GEMINI_LLM`, `GEMINI_EMBED` | defaults de la librería | Modelos de Gemini |

**Una tabla por proveedor.** La columna de embeddings tiene dimensión fija, y los vectores de modelos distintos no son comparables. Cambiar de `PROVIDER` usa otra tabla, así que podés alternar sin reindexar. Si cambiás el modelo de embeddings dentro del mismo proveedor, usá otro `PG_TABLE`. Si la dimensión no coincide con la de la tabla, la app se detiene con un mensaje claro.

## Proveedores

**Ollama.**
- `OLLAMA_CONTEXT_WINDOW` evita que LlamaIndex use el contexto máximo del modelo (128k en llama3.1). Con 128k, la memoria de la GPU no alcanza, el modelo se reparte con la CPU y las consultas dan timeout. 8192 alcanza para 4 fragmentos de 512 tokens.
- Para guardar los modelos en otro disco, definí `OLLAMA_MODELS` antes del `pull`.
- Ollama descarga el modelo de memoria tras 5 minutos sin uso. La siguiente pregunta tarda lo que lleve volver a cargarlo: unos 35 segundos desde un disco rígido, menos de un segundo con el modelo cargado.

**Gemini sobre Vertex AI.** En local, autenticate con `gcloud auth application-default login` y definí `GCP_PROJECT`. En Cloud Run, la identidad es la cuenta de servicio.

**Gemini con API key.** `PROVIDER=gemini` y `GOOGLE_API_KEY` de [Google AI Studio](https://aistudio.google.com). Es el mismo modelo, sin proyecto de Google Cloud.

**OpenAI.** `PROVIDER=openai` y `OPENAI_API_KEY`.

## Limitaciones conocidas

- **Idioma de los embeddings.** nomic-embed-text está entrenado casi solo en inglés, y para él coincidir en idioma pesa más que coincidir en tema. Con los papers en inglés y preguntas en español funciona, pero si se indexa un documento en español, este captura las preguntas en español aunque no tenga relación: en una prueba, la evaluación bajó de 6/6 a 3/6. La solución es un modelo de embeddings multilingüe, como `bge-m3` en Ollama. Por la misma razón, algunas preguntas generales se niegan: por ejemplo, "¿Qué problema resuelve RAG frente a un modelo sin recuperación?" responde que no encuentra la información, porque el fragmento que la responde queda quinto.
- **Sin autenticación.** Cualquiera con acceso a la API puede subir documentos y hacer preguntas.
- **Un solo proceso escribiendo.** El candado que evita subidas duplicadas simultáneas vale dentro de un proceso. Con varias instancias haría falta un bloqueo en la base.
- **Tailwind en el navegador** genera el CSS al cargar la página. Para mucho tráfico conviene compilarlo, aunque eso agrega un paso de build.

## Stack

Python · LlamaIndex · FastAPI · PostgreSQL + pgvector · HTMX · Tailwind CSS · Docker · pytest · Ollama · Gemini / Vertex AI · OpenAI · Google Cloud Run · Supabase
