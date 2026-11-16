"""Extraction P/R/F1 against the gold labels (owner: HA).

pred: one row per predicted event (columns: doc_id, trigger, date, category; extra columns ignored).
      A document that has no row is treated as 'no event predicted'.
gold: data/annotations/gold_all_labels_v1.csv (has_event, event_date, category, trigger, source_type, annotator).
"""
from __future__ import annotations

import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

TR = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‑": "-", "‐": "-", " ": " "})


def _norm(s) -> str:
    return re.sub(r"\s+", " ", str(s or "").translate(TR)).strip().lower()


def _trigger_hit(g, p) -> bool:
    g, p = _norm(g), _norm(p)
    if not g or not p:
        return False
    if g in p or p in g:
        return True
    gt, pt = set(g.split()), set(p.split())
    return len(gt & pt) / len(gt | pt) >= 0.5


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def evaluate(pred: pd.DataFrame, gold: pd.DataFrame, exclude_annotators=()) -> pd.DataFrame:
    gold = gold[~gold["annotator"].isin(exclude_annotators)].copy()
    pred = pred.sort_values("doc_id").drop_duplicates("doc_id", keep="first").set_index("doc_id")   # first row = primary event
    rows = []
    for scope, g in [("all", gold)] + [(st, x) for st, x in gold.groupby("source_type")]:
        is_ev = g["has_event"] == "yes"
        has_pred = g["doc_id"].isin(pred.index)
        tp, fp, fn = int((is_ev & has_pred).sum()), int((~is_ev & has_pred).sum()), int((is_ev & ~has_pred).sum())
        p, r, f = prf(tp, fp, fn)
        hit = g[is_ev & has_pred]
        has_trig = "trigger" in pred.columns and (pred["trigger"].astype(str).str.len() > 0).any()
        trig = [_trigger_hit(a, pred.loc[d, "trigger"]) for a, d in zip(hit["trigger"], hit["doc_id"])] if has_trig else []
        day = [str(pred.loc[d, "date"])[:10] == a[:10] for a, d in zip(hit["event_date"], hit["doc_id"])]
        mon = [str(pred.loc[d, "date"])[:7] == a[:7] for a, d in zip(hit["event_date"], hit["doc_id"])]
        cat_true = list(hit["category"])
        cat_pred = [pred.loc[d, "category"] if "category" in pred.columns else "" for d in hit["doc_id"]]
        n = max(len(hit), 1)
        rows.append({
            "scope": scope, "docs": len(g), "gold_events": int(is_ev.sum()),
            "detect_precision": p, "detect_recall": r, "detect_f1": f,
            "trigger_hit": (sum(trig) / n) if has_trig else None, "date_exact_day": sum(day) / n, "date_same_month": sum(mon) / n,
            "category_accuracy": accuracy_score(cat_true, cat_pred) if cat_true else None,
            "category_macro_f1": f1_score(cat_true, cat_pred, average="macro", zero_division=0) if cat_true else None,
        })
    return pd.DataFrame(rows)
