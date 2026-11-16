"""Run the D8 extraction pipeline over a corpus JSONL.

    python scripts/run_extraction_pipeline.py data/interim/corpus_v1_clean.jsonl [--out data/processed] [--limit N]

Writes  <out>/extractions_v1.jsonl  (full records incl. unit text; keep out of public repos)
        <out>/extractions_v1.csv    (flat, no raw text)
        <out>/extraction_summary.txt
"""
from __future__ import annotations
import argparse, csv, json, sys, time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.extraction.pipeline import extract_document

FLAT = ["doc_id", "source", "source_type", "title", "publication_date", "has_candidate_event", "n_units", "trigger",
        "trigger_rule", "trigger_score", "event_date", "event_date_how", "text_date", "organizations", "systems"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("--out", default="data/processed"); ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    recs, t0 = [], time.time()
    for i, line in enumerate(open(a.corpus, encoding="utf-8")):
        if a.limit and i >= a.limit:
            break
        recs.append(extract_document(json.loads(line)))
    with open(out / "extractions_v1.jsonl", "w", encoding="utf-8") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out / "extractions_v1.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FLAT); w.writeheader()
        for r in recs:
            row = {k: r[k] for k in FLAT}
            row["organizations"] = "; ".join(r["organizations"]); row["systems"] = "; ".join(r["systems"])
            w.writerow(row)
    n = len(recs)
    lines = [f"docs: {n}   time: {time.time() - t0:.1f}s"]
    for st in ("lab_blog", "paper", "release_notes"):
        s = [r for r in recs if r["source_type"] == st]
        if not s:
            continue
        c = lambda f: sum(1 for r in s if f(r))
        lines.append(f"{st:14} n={len(s):4}  candidate_event={c(lambda r: r['has_candidate_event'])/len(s):.0%}  "
                     f"text_date_found={c(lambda r: r['text_date'])/len(s):.0%}  "
                     f"org={c(lambda r: r['organizations'])/len(s):.0%}  system={c(lambda r: r['systems'])/len(s):.0%}")
    lines.append("trigger rules: " + ", ".join(f"{k}={v}" for k, v in Counter(r["trigger_rule"] for r in recs if r["trigger_rule"]).most_common()))
    lines.append("event_date_how: " + ", ".join(f"{k}={v}" for k, v in Counter(r["event_date_how"] for r in recs).most_common()))
    (out / "extraction_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
