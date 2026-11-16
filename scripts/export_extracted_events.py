"""Export extracted events in the event schema for loading with origin=extracted (gold stays the default view).

    python scripts/export_extracted_events.py data/interim/corpus_v1_clean.jsonl gold_events_v1_1.csv gold_all_labels_v1.csv \
        [--thresholds data/processed/thresholds.json | --threshold 0.8] [--out data/processed/extracted_events.csv]

Gold documents are excluded (they are already in the gold view). confidence = p_event. impact is empty (not extracted).
Without --thresholds a single --threshold is used and the output is PROVISIONAL.
"""
from __future__ import annotations
import argparse, csv, json, sys, warnings
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import load_corpus, load_csv
from ainevo.classification.events_export import SCHEMA, event_row, fit_models, p_event

warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("alllabels")
    ap.add_argument("--thresholds"); ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--out", default="data/processed/extracted_events.csv")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]; al = [r for r in load_csv(a.alllabels) if r["doc_id"] in docs]
    gold = {r["doc_id"] for r in al}
    thr = json.loads(Path(a.thresholds).read_text()) if a.thresholds else None
    models = fit_models(docs, ev, al)
    ids = [i for i in docs if i not in gold]
    p = p_event(models, docs, ids)
    rows = []
    for i, pv in zip(ids, p):
        d = docs[i]
        t = thr.get(d["source_type"], thr["global"]) if thr else a.threshold
        if pv >= t:
            rows.append(event_row(models, d, float(pv), len(rows) + 1))
    rows.sort(key=lambda r: (r["date"] or "", r["event_id"]))
    for n, r in enumerate(rows, 1):
        r["event_id"] = f"EX-{n:05d}"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=SCHEMA); w.writeheader(); w.writerows(rows)
    print(f"{'thresholds ' + json.dumps({k: v for k, v in thr.items() if k != 'min_precision'}) if thr else 'PROVISIONAL threshold ' + str(a.threshold)}")
    print(f"non-gold docs scored: {len(ids)}   exported events: {len(rows)}   by source: {dict(Counter(r['source_type'] for r in rows))}")
    print("by category:", dict(Counter(r['category'] for r in rows).most_common()))


if __name__ == "__main__":
    main()
