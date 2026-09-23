# RAG sobre documentos propios

Asistente que responde preguntas usando **solo** el contenido de tus documentos (PDFs, .txt, .md, .docx), citando de qué archivo y página sale cada respuesta. Si la información no está en los documentos, lo dice en vez de inventar.

Funciona con **Gemini sobre Vertex AI (GCP)**, **Gemini** vía API key, **OpenAI** u **Ollama** (modelos locales: gratis y sin que los datos salgan de tu máquina). Se cambia con una variable.

## Cómo funciona

```
Documentos ──► Chunking ──► Embeddings ──► Índice vectorial (persistido en disco)
                                                    │
Pregunta ──► Embedding ──► Búsqueda top-k ──────────┘
                                │
                                ▼
                  Prompt con contexto recuperado ──► LLM ──► Respuesta + fuentes
```

1. **Ingesta:** se leen los documentos y se parten en fragmentos (`SentenceSplitter`, 512 tokens con solapamiento de 64 para no cortar ideas a la mitad).
2. **Indexación:** cada fragmento se convierte en un embedding y se guarda en un índice vectorial en `storage/`.
3. **Consulta:** la pregunta se convierte en embedding, se recuperan los 4 fragmentos más similares y se le pasan al modelo con un prompt que lo obliga a responder solo con ese contexto.

## Inicio rápido

```bash
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # elegí PROVIDER y completá la clave si hace falta
```

> En Windows, corré los comandos con `set PYTHONUTF8=1` (o `$env:PYTHONUTF8=1` en PowerShell): la consola por defecto usa cp1252 y no puede imprimir algunos símbolos de las respuestas ni los ✓/✗ de la evaluación.

Copiá tus documentos a `data/` y:

```bash
python rag.py ingest                              # indexa
python rag.py ask "¿Cada cuánto se hace el mantenimiento?"
python rag.py chat                                # modo interactivo
```

Ejemplo de salida:

```
El mantenimiento preventivo se realiza cada 6 meses.

Fuentes:
  · manual_equipo.pdf (pág. 12) — similitud 0.84
  · procedimiento_mantenimiento.pdf (pág. 3) — similitud 0.79
```

## Documentos indexados

En `data/` van cuatro papers de arXiv (incluidos en el repo), que sirven como corpus de ejemplo:

| Archivo | Paper | Páginas |
|---|---|---|
| `Attention_Is_All_You_Need.pdf` | [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) | 15 |
| `Retrieval-Augmented_Generation_for_Knowledge-Intensive_NLP_Tasks.pdf` | [RAG for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) (Lewis et al., 2020) | 19 |
| `ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.pdf` | [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629) (Yao et al., 2022) | 33 |
| `Lost_in_the_Middle_How_Language_Models_Use_Long_Contexts.pdf` | [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172) (Liu et al., 2023) | 18 |

Con la configuración por defecto (`CHUNK_SIZE=512`, `CHUNK_OVERLAP=64`) el ingest produce **85 páginas en 228 chunks**. El índice (`storage/`) no va al repo: se regenera con `python rag.py ingest`.

Ejemplo real con `PROVIDER=ollama` (llama3.1 + nomic-embed-text):

```
$ python rag.py ask "¿Qué problema resuelve RAG frente a un modelo sin recuperación?"

RAG es capaz de actualizar su conocimiento del mundo simplemente reemplazando su memoria
no paramétrica, lo que le permite mejorar su precisión en tareas de recuperación y generación.

Fuentes:
  · ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.pdf (pág. 28) — similitud 0.59
  · Retrieval-Augmented_Generation_for_Knowledge-Intensive_NLP_Tasks.pdf (pág. 8) — similitud 0.54
  · ReAct_Synergizing_Reasoning_and_Acting_in_Language_Models.pdf (pág. 28) — similitud 0.54
  · Retrieval-Augmented_Generation_for_Knowledge-Intensive_NLP_Tasks.pdf (pág. 7) — similitud 0.52
```

## Modo local con Ollama

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

En `.env`: `PROVIDER=ollama`, y volvé a correr `python rag.py ingest` (los embeddings de cada proveedor no son compatibles entre sí: cada vez que cambies de `PROVIDER`, reindexá).

Notas:

- **`OLLAMA_CONTEXT_WINDOW`** (default `8192`): si no se fija, LlamaIndex usa el contexto máximo del modelo (128k en llama3.1). Con eso la caché KV no entra en una GPU de 8 GB, el modelo se reparte con la CPU y las consultas dan timeout. 8192 alcanza de sobra para `TOP_K=4` chunks de 512 tokens.
- **Modelos en otro disco:** Ollama guarda los modelos en `%USERPROFILE%\.ollama\models`. Para moverlos, definí la variable de entorno `OLLAMA_MODELS` (por ejemplo `D:\ollama\models`) antes de hacer el `pull`. Conviene que sea un SSD: la primera consulta después de un rato de inactividad tiene que volver a cargar los ~5 GB de llama3.1 en memoria.

## Gemini sobre Vertex AI (Google Cloud)

1. Creá un proyecto en [console.cloud.google.com](https://console.cloud.google.com) (las cuentas nuevas tienen crédito gratis).
2. En el proyecto, habilitá la **Vertex AI API** (APIs y servicios → Biblioteca → "Vertex AI API").
3. Instalá la [gcloud CLI](https://cloud.google.com/sdk/docs/install) y autenticate:

```bash
gcloud auth application-default login
gcloud config set project TU_PROYECTO
```

4. En `.env`:

```
PROVIDER=vertex
GCP_PROJECT=TU_PROYECTO
GCP_LOCATION=us-central1
```

5. `python rag.py ingest` y listo. No hace falta API key: la autenticación la toma de las credenciales de gcloud (Application Default Credentials), que es como se trabaja en entornos de GCP.

**Alternativa sin GCP:** con una API key de [Google AI Studio](https://aistudio.google.com), `PROVIDER=gemini` y `GOOGLE_API_KEY=...`. Mismo modelo, sin proyecto de nube.

Los modelos se pueden fijar con `GEMINI_LLM` y `GEMINI_EMBED`; si quedan vacíos usa los default de la librería.

## Evaluación

`evaluate.py` corre un set de preguntas de `eval_set.json` y mide dos cosas:

- **Aciertos:** la respuesta contiene los datos esperados.
- **Negativas correctas:** ante preguntas fuera de los documentos, el sistema se niega en vez de alucinar.

```bash
python evaluate.py
```

`eval_set.json` trae 6 preguntas sobre los papers de `data/`: cuatro con respuesta en los documentos y dos fuera de alcance (capital de Australia, Mundial 2022) para verificar que el sistema se niega en vez de inventar. Si cambiás los documentos, reemplazá las preguntas. Sirve para comparar configuraciones (tamaño de chunk, `TOP_K`, un proveedor contra otro) con números en vez de impresiones.

### Resultados

`PROVIDER=ollama` (llama3.1 + nomic-embed-text), `CHUNK_SIZE=512`, `CHUNK_OVERLAP=64`, `TOP_K=4`, en una RTX 3070 (8 GB):

| # | Pregunta | Esperado | Resultado |
|---|---|---|---|
| 1 | ¿Cuántas cabezas de atención usa el Transformer base? | `8` | ✓ |
| 2 | ¿Cuáles son las dos variantes de RAG y en qué se diferencian? | `Sequence`, `Token` | ✓ |
| 3 | ¿En qué dos benchmarks de razonamiento se evalúa ReAct? | `HotpotQA`, `FEVER` | ✓ |
| 4 | ¿Qué forma tiene la curva de rendimiento según la posición del contexto (Lost in the Middle)? | `U` | ✓ |
| 5 | ¿Cuál es la capital de Australia? | se niega | ✓ |
| 6 | ¿Qué equipo ganó la Copa del Mundo 2022? | se niega | ✓ |

**6/6 (100%)**, 37 s en total con el modelo ya cargado en memoria (unos 6 s por pregunta).

## Configuración

| Variable | Default | Qué controla |
|---|---|---|
| `PROVIDER` | `openai` | `vertex`, `gemini`, `openai` u `ollama` |
| `CHUNK_SIZE` | `512` | Tamaño de cada fragmento (tokens) |
| `CHUNK_OVERLAP` | `64` | Solapamiento entre fragmentos |
| `TOP_K` | `4` | Fragmentos recuperados por pregunta |
| `OLLAMA_CONTEXT_WINDOW` | `8192` | Contexto (`num_ctx`) del LLM en Ollama; ver notas en "Modo local con Ollama" |

## Stack

Python · LlamaIndex · Gemini / Vertex AI (GCP) · OpenAI API · Ollama · embeddings semánticos
