"""Segment documents into event units and write a CSV (no document text, only ids/offsets/trigger/score).
Usage: python scripts/run_event_segmentation.py <in.jsonl> <out.csv> [--min-score 3.0] [--max-units 5]
"""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.extraction.segment import segment_events

args = sys.argv[1:]
src, out = args[0], args[1]
ms = float(args[args.index("--min-score") + 1]) if "--min-score" in args else 3.0
mu = int(args[args.index("--max-units") + 1]) if "--max-units" in args else 5
rows = []; n_docs = 0
for line in open(src, encoding="utf-8"):
    d = json.loads(line); n_docs += 1
    for u in segment_events(d["text"], d["source_type"], min_score=ms, max_units=mu):
        rows.append({"doc_id": d["doc_id"], "source_type": d["source_type"], "publication_date": d.get("publication_date"),
                     "unit_id": u["unit_id"], "rank": u["rank"], "is_primary": u["is_primary"], "start": u["start"], "end": u["end"],
                     "n_sentences": u["n_sentences"], "trigger": u["trigger"], "rule": u["rule"], "score": u["score"],
                     "section": u["section"], "dates": ";".join(u["dates"])})
fields = list(rows[0]) if rows else ["doc_id"]
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
print(f"{n_docs} docs -> {len(rows)} units ({len({r['doc_id'] for r in rows})} docs with >=1 unit) -> {out}")
