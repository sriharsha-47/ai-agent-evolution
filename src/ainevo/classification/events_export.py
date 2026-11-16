"""Shared helpers: fit the has_event / attribute models on gold and build event-schema rows for non-gold documents."""
from __future__ import annotations
import numpy as np

from ainevo.classification.baseline import doc_text, make_model
from ainevo.extraction.relations import extract_relation, rule_capability
from ainevo.extraction.pipeline import extract_document

SCHEMA = ["event_id", "doc_id", "date", "category", "trigger", "organization", "model_agent", "capability", "action",
          "autonomy_level", "interaction", "impact", "source", "source_type", "confidence", "origin"]


def fit_models(docs: dict, events: list[dict], alllabels: list[dict]) -> dict:
    T = lambda r: doc_text(docs[r["doc_id"]])
    return {
        "event": make_model().fit([T(r) for r in alllabels], ["yes" if r["has_event"] == "yes" else "no" for r in alllabels]),
        "category": make_model().fit([T(r) for r in events], [r["category"] for r in events]),
        "autonomy": make_model().fit([T(r) for r in events], [r["autonomy_level"] for r in events]),
        "interaction": make_model().fit([T(r) for r in events], [r["interaction"] for r in events]),
    }


def p_event(models: dict, docs: dict, ids: list[str]) -> np.ndarray:
    m = models["event"]
    return m.predict_proba([doc_text(docs[i]) for i in ids])[:, list(m.classes_).index("yes")]


def event_row(models: dict, doc: dict, p: float, n: int) -> dict:
    X = [doc_text(doc)]
    rec = extract_document(doc)
    rel = extract_relation(doc)
    return {"event_id": f"EX-{n:05d}", "doc_id": doc["doc_id"], "date": doc.get("publication_date"),
            "category": models["category"].predict(X)[0], "trigger": rec["trigger"] or "",
            "organization": "; ".join(rec["organizations"]), "model_agent": rec["systems"][0] if rec["systems"] else "",
            "capability": rule_capability(doc), "action": rel["action"],
            "autonomy_level": models["autonomy"].predict(X)[0], "interaction": models["interaction"].predict(X)[0],
            "impact": "", "source": doc.get("source"), "source_type": doc.get("source_type"),
            "confidence": f"{p:.3f}", "origin": "extracted"}
