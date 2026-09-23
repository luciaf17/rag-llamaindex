"""Evaluación mínima del RAG.

Editá eval_set.json con preguntas sobre TUS documentos:
  - "expected": palabras que la respuesta correcta debería contener
  - "out_of_scope": true si la respuesta NO está en los documentos
    (el sistema debería negarse en vez de inventar)

Uso: python evaluate.py
"""
import json

import config
from rag import NO_ANSWER, build_query_engine, load_index


def main() -> None:
    config.configure_models()
    engine = build_query_engine(load_index())

    with open("eval_set.json", encoding="utf-8") as f:
        cases = json.load(f)

    passed = 0
    for i, case in enumerate(cases, 1):
        answer = str(engine.query(case["question"])).strip()
        refused = NO_ANSWER.lower() in answer.lower()

        if case.get("out_of_scope"):
            ok = refused
            detail = "se negó correctamente" if ok else "inventó una respuesta"
        else:
            missing = [w for w in case.get("expected", []) if w.lower() not in answer.lower()]
            ok = not refused and not missing
            detail = "ok" if ok else (f"faltan: {missing}" if not refused else "se negó sin motivo")

        passed += ok
        print(f"[{'✓' if ok else '✗'}] {i}. {case['question']}\n     → {detail}")

    print(f"\nResultado: {passed}/{len(cases)} ({passed / len(cases):.0%})")


if __name__ == "__main__":
    main()
