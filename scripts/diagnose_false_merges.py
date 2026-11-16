"""Diagnose the reviewed false merges (link_false_merges.csv from Harshita) against the extraction fields.

    python scripts/diagnose_false_merges.py data/interim/corpus_v1_clean.jsonl link_false_merges.csv [--out results/tables]

For each reviewed pair (all labelled N = not the same event) it shows whether org + system (+ date window) alone would still
merge them, with the old system names (as in the file) and the current ones (hyphen-normalised, o1 family fixed), and how well a
title-similarity gate separates them. Only false merges are in the file, so this CANNOT choose a gate threshold: a sample of
true duplicates is needed for that.
"""
from __future__ import annotations
import argparse, csv, sys
from collections import Counter
from datetime import date
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import load_corpus
from ainevo.extraction.ner import canon, guess_organizations, guess_systems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("pairs"); ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    rows = [r for r in csv.DictReader(open(a.pairs, encoding="utf-8-sig")) if r["event_group"]]
    lines = [f"pairs: {len(rows)}  groups: {len({r['event_group'] for r in rows})}  docs: {len({r['doc_a'] for r in rows} | {r['doc_b'] for r in rows})}  "
             f"labels: {dict(Counter(r['same_event? (Y/N)'] for r in rows))}"]
    gap = lambda r: abs((date.fromisoformat(r["date_a"]) - date.fromisoformat(r["date_b"])).days)
    lines.append(f"date gap (days): {dict(sorted(Counter(gap(r) for r in rows).items()))}")
    old_same = sum(canon(r["model_agent_a"]).lower() == canon(r["model_agent_b"]).lower() for r in rows)
    lines.append(f"same system with the names in the file (hyphen-normalised): {old_same}/{len(rows)}; "
                 f"raw U+2011 in file names: {sum(chr(0x2011) in r['model_agent_a'] + r['model_agent_b'] for r in rows)}/{len(rows)}; "
                 f"generic 'ChatGPT' on either side: {sum('chatgpt' in (r['model_agent_a'] + r['model_agent_b']).lower() for r in rows)}")
    first = lambda i: (guess_systems(docs[i]) or [""])[0].lower()
    new_same = [r for r in rows if first(r["doc_a"]) and first(r["doc_a"]) == first(r["doc_b"])]
    lines.append(f"same system with CURRENT extraction (first system): {len(new_same)}/{len(rows)}  -> separated by the name fix alone: {len(rows) - len(new_same)}")
    v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit([d["title"] for d in docs.values()])
    sim = lambda r: float(v.transform([docs[r["doc_a"]]["title"]]).multiply(v.transform([docs[r["doc_b"]]["title"]])).sum())
    s = np.array([sim(r) for r in rows])
    lines.append("title TF-IDF cosine of these false pairs: median %.2f, 90th pct %.2f, max %.2f" % (np.median(s), np.percentile(s, 90), s.max()))
    for t in (0.2, 0.3, 0.5):
        lines.append(f"  a title-similarity gate >= {t} would still merge {int((s >= t).sum())}/{len(rows)} of these false pairs")
    lines.append("pairs still merged by the name fix AND a 0.3 title gate: %d" % sum(1 for r, x in zip(rows, s) if x >= .3 and first(r["doc_a"]) == first(r["doc_b"])))
    txt = "\n".join(lines); print(txt)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "false_merge_diagnosis.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
