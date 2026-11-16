"""Extraction-derived features for the classifier (from ainevo.extraction.pipeline records)."""
from __future__ import annotations
import numpy as np

from ainevo.extraction.pipeline import extract_document

RULES = ["announce", "research", "feature", "model", "deal", "scale", "describe"]
SOURCES = ["lab_blog", "paper", "release_notes"]
FEATURE_NAMES = (["has_trigger", "trigger_score", "n_units", "has_system", "has_org", "n_systems", "log_len"]
                 + [f"rule_{r}" for r in RULES] + [f"src_{s}" for s in SOURCES])


def feature_vector(doc: dict) -> list[float]:
    r = extract_document(doc)
    rule = r["trigger_rule"]
    return ([1.0 if r["has_candidate_event"] else 0.0, float(r["trigger_score"] or 0.0) / 10.0, float(r["n_units"]) / 5.0,
             1.0 if r["systems"] else 0.0, 1.0 if r["organizations"] else 0.0, float(len(r["systems"])),
             float(np.log1p(len(doc.get("text") or ""))) / 10.0]
            + [1.0 if rule == x else 0.0 for x in RULES]
            + [1.0 if doc.get("source_type") == s else 0.0 for s in SOURCES])
