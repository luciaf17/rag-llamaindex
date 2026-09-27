"""Fragmentos HTML que devuelve la interfaz HTMX.

HTMX pide al servidor pedazos de HTML ya armados y los inserta en la página,
en vez de pedir JSON y armarlo con JavaScript. Estas funciones arman esos
pedazos. Todo texto que viene de afuera (respuestas del modelo, nombres de
archivo, preguntas) pasa por escape(): si no, un archivo llamado
<script>...</script>.pdf ejecutaría código en el navegador (XSS).
"""
import json
from html import escape

CARD = "rounded-lg px-4 py-3 text-sm leading-relaxed"


def _chunks(n: int) -> str:
    return f"{n} chunk" if n == 1 else f"{n} chunks"


def documents(docs: list[dict]) -> str:
    if not docs:
        return '<p class="text-sm text-slate-500">Todavía no hay documentos. Subí uno arriba.</p>'
    items = []
    for d in docs:
        name = escape(d["file_name"])
        pages = f"{d['pages']} págs · " if d["pages"] else ""
        items.append(
            f'<li class="py-2">'
            f'<p class="truncate text-sm font-medium" title="{name}">{name}</p>'
            f'<p class="text-xs text-slate-500">{pages}{_chunks(d["chunks"])}</p>'
            f"</li>"
        )
    return f'<ul class="divide-y divide-slate-200 dark:divide-slate-800">{"".join(items)}</ul>'


def upload_ok(doc: dict) -> str:
    return (
        f'<p class="text-emerald-700 dark:text-emerald-400">'
        f'✓ {escape(doc["file_name"])} indexado ({_chunks(doc["chunks"])})</p>'
    )


def upload_error(message: str) -> str:
    return f'<p class="text-red-700 dark:text-red-400">{escape(message)}</p>'


def question(text: str) -> str:
    """La pregunta del usuario y un lugar para la respuesta que se carga solo.

    El div de "Pensando…" tiene hx-trigger="load": apenas HTMX lo inserta en
    la página, dispara el POST a /ui/answer y se reemplaza con la respuesta.
    Así la pregunta aparece al instante aunque el modelo tarde.
    """
    vals = escape(json.dumps({"question": text}), quote=True)
    return (
        f'<div class="flex justify-end">'
        f'<p class="{CARD} max-w-[85%] whitespace-pre-line bg-indigo-600 text-white">{escape(text)}</p>'
        f"</div>"
        f'<div hx-post="/ui/answer" hx-trigger="load" hx-vals="{vals}" hx-swap="outerHTML">'
        f'<p class="{CARD} inline-block animate-pulse bg-slate-100 text-slate-500 dark:bg-slate-800">'
        f"Pensando…</p>"
        # Aparece a los 8 s (animación "appear" en index.html) si todavía no hay respuesta.
        f'<p class="mt-1 text-xs text-slate-400 opacity-0 [animation:appear_0.3s_8s_forwards]">'
        f"La primera respuesta puede tardar hasta un minuto si el modelo se está cargando.</p>"
        f"</div>"
    )


def _dedupe(sources: list[dict]) -> list[dict]:
    """Un mismo archivo y página puede aparecer en varios chunks: se muestra una vez."""
    best: dict[tuple, dict] = {}
    for s in sources:
        key = (s["file_name"], s["page"])
        if key not in best or (s["score"] or 0) > (best[key]["score"] or 0):
            best[key] = s
    return sorted(best.values(), key=lambda s: -(s["score"] or 0))


def answer(text: str, sources: list[dict]) -> str:
    rows = []
    for s in _dedupe(sources):
        if s["score"] is None:
            continue
        page = f" · pág. {escape(str(s['page']))}" if s["page"] else ""
        rows.append(
            f'<li class="flex justify-between gap-3">'
            f'<span class="truncate">{escape(s["file_name"])}{page}</span>'
            f'<span class="shrink-0 tabular-nums">{s["score"]:.2f}</span>'
            f"</li>"
        )
    items = "".join(rows)
    sources_html = (
        f'<div class="mt-3 border-t border-slate-200 pt-2 dark:border-slate-700">'
        f'<p class="mb-1 text-xs font-semibold uppercase tracking-wide text-slate-500">Fuentes</p>'
        f'<ul class="space-y-0.5 text-xs text-slate-600 dark:text-slate-400">{items}</ul>'
        f"</div>"
        if items
        else ""
    )
    return (
        f'<div class="flex justify-start">'
        f'<div class="{CARD} max-w-[85%] bg-slate-100 dark:bg-slate-800">'
        f'<p class="whitespace-pre-line">{escape(text)}</p>{sources_html}'
        f"</div></div>"
    )


def answer_error(message: str) -> str:
    return (
        f'<div class="flex justify-start">'
        f'<p class="{CARD} max-w-[85%] bg-red-50 text-red-800 dark:bg-red-950 dark:text-red-300">'
        f"{escape(message)}</p></div>"
    )
