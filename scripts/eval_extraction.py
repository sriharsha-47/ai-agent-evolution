"""Evaluate Harsha's extracted events against the gold labels.

Usage (repo root):
  python scripts/eval_extraction.py --pred data/processed/events.jsonl --exclude-annotators SH
(--exclude-annotators SH keeps out the documents Harsha tuned his rules on; the number is then less optimistic.)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ainevo.evaluation.extraction_eval import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help=".csv or .jsonl with doc_id, trigger, date, category")
    ap.add_argument("--gold", default=str(ROOT / "data" / "annotations" / "gold_all_labels_v1.csv"))
    ap.add_argument("--exclude-annotators", nargs="*", default=[])
    ap.add_argument("--triggers", default=None, help="optional extractions_v1.csv to take trigger text from")
    ap.add_argument("--min-p", type=float, default=None, help="keep only predicted events with p_event >= this value")
    a = ap.parse_args()
    pred = pd.read_json(a.pred, lines=True) if a.pred.endswith(".jsonl") else pd.read_csv(a.pred, dtype=str, keep_default_na=False)
    # Harsha's final file: event_date / category_pred / has_event_pred / p_event (no trigger column)
    pred = pred.rename(columns={"event_date": "date", "category_pred": "category"})
    if a.triggers:
        trg = pd.read_csv(a.triggers, dtype=str, keep_default_na=False)[["doc_id", "trigger"]].drop_duplicates("doc_id")
        pred = pred.drop(columns=[c for c in ["trigger"] if c in pred.columns]).merge(trg, on="doc_id", how="left")
    if "has_event_pred" in pred.columns:
        pred = pred[pred["has_event_pred"].str.lower() == "yes"]
        if a.min_p is not None and "p_event" in pred.columns:
            pred = pred[pred["p_event"].astype(float) >= a.min_p]
    if "trigger" not in pred.columns:
        pred["trigger"] = ""
    missing = [c for c in ("doc_id", "trigger", "date") if c not in pred.columns]
    if missing:
        raise SystemExit(f"{a.pred} lacks columns {missing}; ask Harsha for doc_id, trigger, date (and category).")
    gold = pd.read_csv(a.gold, dtype=str, keep_default_na=False)
    res = evaluate(pred, gold, a.exclude_annotators)
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    res.to_csv(ROOT / "results" / "metrics" / "extraction_eval.csv", index=False)
    print(res.round(3).to_string(index=False))
    print("\nwrote results/metrics/extraction_eval.csv")
    print("NOTE: documents with in_gold=True were used to train the classifier, so detection scores on them are optimistic.")
    print("Quote Harsha's cross-validation numbers for the classifier and use this table for trigger/date agreement.")


if __name__ == "__main__":
    main()
