"""Choose the has_event threshold per source type from a labelled FRESH sample, write thresholds.json.

    python scripts/choose_threshold.py data/interim/corpus_v1_clean.jsonl gold_events_v1_1.csv gold_all_labels_v1.csv threshold_sample.xlsx [--min-precision 0.85]

Models are fitted on gold only, so the sample must not overlap gold (make_threshold_sample.py guarantees that).
Per source type: the lowest threshold whose precision on the sample is >= --min-precision (maximises recall at that precision);
if none reaches it, the threshold with the best F1. Sources with fewer than 5 labelled events fall back to the global choice.
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import load_corpus, load_csv
from ainevo.classification.events_export import fit_models, p_event

warnings.filterwarnings("ignore")
GRID = [round(x, 2) for x in np.arange(0.30, 0.96, 0.05)]


def prf(y, p):
    tp = int((y & p).sum()); pr = tp / p.sum() if p.sum() else 0.0; rc = tp / y.sum() if y.sum() else 0.0
    return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)


def pick(y, s, min_prec):
    rows = [(t,) + prf(y, s >= t) for t in GRID]
    ok = [r for r in rows if r[1] >= min_prec and (s >= r[0]).sum() > 0]
    return (ok[0] if ok else max(rows, key=lambda r: r[3])), rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("alllabels"); ap.add_argument("sample")
    ap.add_argument("--min-precision", type=float, default=0.85); ap.add_argument("--out", default="data/processed/thresholds.json")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]; al = [r for r in load_csv(a.alllabels) if r["doc_id"] in docs]
    ws = openpyxl.load_workbook(a.sample, data_only=True)["Sample"]
    h = [c.value for c in ws[1]]
    rows = [dict(zip(h, r)) for r in ws.iter_rows(min_row=2, values_only=True) if r[h.index("has_event")]]
    gold = {r["doc_id"] for r in al}
    assert not any(r["doc_id"] in gold for r in rows), "sample overlaps gold"
    models = fit_models(docs, ev, al)
    ids = [r["doc_id"] for r in rows]
    s = p_event(models, docs, ids); y = np.array([r["has_event"] == "yes" for r in rows]); st = np.array([r["source_type"] for r in rows])
    print(f"labelled sample: {len(rows)} rows, yes-rate {y.mean():.2f}; by source " + ", ".join(f"{k}={y[st==k].mean():.2f} (n={int((st==k).sum())})" for k in sorted(set(st))))
    g, _ = pick(y, s, a.min_precision)
    out = {"min_precision": a.min_precision, "global": g[0]}
    print(f"global: threshold {g[0]:.2f}  P={g[1]:.2f} R={g[2]:.2f} F1={g[3]:.2f}")
    for k in sorted(set(st)):
        m = st == k
        if y[m].sum() < 5:
            out[k] = g[0]; print(f"{k}: <5 events in sample, using global {g[0]:.2f}"); continue
        r, _ = pick(y[m], s[m], a.min_precision)
        out[k] = r[0]; print(f"{k}: threshold {r[0]:.2f}  P={r[1]:.2f} R={r[2]:.2f} F1={r[3]:.2f}  (events {int(y[m].sum())}/{int(m.sum())})")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
