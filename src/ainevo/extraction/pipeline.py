"""D8: per-document extraction pipeline (no spaCy, no network, deterministic).

Chains the Task 1-5 pieces: event-unit segmentation (triggers), event date, organisations and system names.
`extract_document(doc)` returns one record; `has_candidate_event` only says "a trigger unit was found",
it is NOT a has_event classifier (trigger score does not separate yes/no_* well, see task2/task3 findings).
"""
from __future__ import annotations

from .ner import guess_organizations, guess_systems
from .segment import segment_events
from .temporal import best_event_date, regex_dates


def extract_document(doc: dict, max_units: int = 5) -> dict:
    text = doc.get("text") or ""
    st = doc.get("source_type")
    units = segment_events(text, source_type=st, max_units=max_units)
    primary = next((u for u in units if u.get("is_primary")), units[0] if units else None)
    pub = doc.get("publication_date")
    # Evaluation on the v1 gold (non-VV rows): gold date == publication_date in 178/179 rows, and text-derived dates
    # lowered accuracy (172/179). So event_date = publication_date; the best in-text date is kept as `text_date`.
    unit_dates = regex_dates(primary["text"]) if primary else []
    text_date, how = best_event_date(unit_dates, None)
    if how != "text":
        text_date, how = best_event_date(regex_dates(text), None)
    text_date = text_date if how == "text" else None
    date, how = (pub, "publication_date") if pub else (text_date, "text" if text_date else "none")
    return {
        "doc_id": doc["doc_id"], "source": doc.get("source"), "source_type": st, "title": doc.get("title"),
        "publication_date": pub, "has_candidate_event": primary is not None, "n_units": len(units),
        "trigger": primary["trigger"] if primary else None, "trigger_rule": primary["rule"] if primary else None,
        "trigger_score": primary["score"] if primary else None,
        "event_date": date, "event_date_how": how, "text_date": text_date,
        "organizations": guess_organizations(doc), "systems": guess_systems(doc),
        "units": [{k: u[k] for k in ("unit_id", "start", "end", "trigger", "rule", "score", "section", "rank", "is_primary", "text")}
                  for u in units],
    }
