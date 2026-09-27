"""Tests de la interfaz web: la página y los fragmentos HTML que pide HTMX."""
from tests.test_api import HORARIOS, VACACIONES, upload


def test_index_page_loads_htmx_and_tailwind(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "htmx.org@2" in response.text
    assert "@tailwindcss/browser@4" in response.text


def test_documents_fragment_lists_uploads(client):
    assert "Todavía no hay documentos" in client.get("/ui/documents").text
    upload(client, "vacaciones.txt", VACACIONES)
    assert "vacaciones.txt" in client.get("/ui/documents").text


def test_upload_fragment_triggers_list_refresh(client):
    response = client.post(
        "/ui/documents", files={"file": ("vacaciones.txt", VACACIONES.encode(), "text/plain")}
    )
    assert response.status_code == 200
    assert response.headers["HX-Trigger"] == "documents-changed"
    assert "vacaciones.txt indexado" in response.text


def test_upload_fragment_keeps_error_status(client):
    response = client.post("/ui/documents", files={"file": ("datos.csv", b"a,b", "text/csv")})
    assert response.status_code == 415
    assert "Formato no soportado" in response.text
    assert "HX-Trigger" not in response.headers


def test_ask_fragment_shows_question_and_loads_answer(client):
    response = client.post("/ui/ask", data={"question": "¿Cuántos días?"})
    assert response.status_code == 200
    assert "¿Cuántos días?" in response.text
    assert 'hx-post="/ui/answer"' in response.text
    assert 'hx-trigger="load"' in response.text


def test_empty_question_adds_nothing(client):
    response = client.post("/ui/ask", data={"question": "   "})
    assert response.status_code == 204


def test_answer_fragment_with_sources(client):
    upload(client, "vacaciones.txt", VACACIONES)
    upload(client, "horarios.txt", HORARIOS)
    response = client.post(
        "/ui/answer", data={"question": "¿Cuántos días hábiles de vacaciones tiene cada empleado?"}
    )
    assert response.status_code == 200
    assert "21 días hábiles" in response.text
    assert "Fuentes" in response.text
    assert "vacaciones.txt" in response.text


def test_answer_fragment_without_documents_shows_error(client):
    response = client.post("/ui/answer", data={"question": "hola"})
    assert response.status_code == 409
    assert "No hay documentos indexados" in response.text


def test_user_text_is_escaped(client):
    """Una pregunta con HTML no se inserta como HTML (evita XSS)."""
    response = client.post("/ui/ask", data={"question": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
