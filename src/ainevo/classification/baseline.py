"""D11 baseline: TF-IDF + logistic regression for event category and has_event (no GPU, no downloads)."""
from __future__ import annotations
import csv, json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

MAX_CHARS = 3000  # annotators read a ~600-char excerpt; the first 3000 chars of the document cover it and more


def load_csv(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_corpus(path: str) -> dict[str, dict]:
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out[d["doc_id"]] = d
    return out


def doc_text(doc: dict, with_source: bool = True) -> str:
    """title + opening of the document (+ a source-type token). Never uses any gold column."""
    s = f"{doc['source_type']} " if with_source else ""
    return s + (doc.get("title") or "") + ". " + (doc.get("text") or "")[:MAX_CHARS]


def make_model(C: float = 10.0):
    return make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode", lowercase=True),
        LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
    )
