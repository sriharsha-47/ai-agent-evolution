"""D16: robustness - per-source and per-period metrics for extraction and classification.

    python scripts/eval_robustness.py data/interim/corpus_v1_clean.jsonl <gold_events_v1_1.csv> <gold_all_labels_v1.csv> [--out results/tables]

Classification numbers are out-of-fold (5-fold, seed 0; TF-IDF + LR), so every document is predicted by a model that never saw it.
Extraction numbers use the rule extractors (D5/D8), which are not fitted on gold. Slices with n < 15 are flagged (*): read them as anecdotes.
Period = publication year bucket (<=2021 | 2022 | 2023 | 2024 | 2025 | 2026).
"""
from __future__ import annotations
import argparse, csv, re, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import doc_text, load_corpus, load_csv, make_model
from ainevo.extraction.ner import guess_organizations, guess_systems

warnings.filterwarnings("ignore")
MIN_N = 15


def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())
def match(a, b): x, y = norm(a), norm(b); return len(x) >= 2 and len(y) >= 2 and (x in y or y in x)
def vals(v): return [p.strip() for p in re.split(r";", v or "") if p.strip()]


def period(d):
    y = int(d[:4]); return "<=2021" if y <= 2021 else str(y)


def oof(texts, y, seed=0):
    p = np.empty(len(y), dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y):
        p[te] = make_model().fit([texts[i] for i in tr], y[tr]).predict([texts[i] for i in te])
    return p.astype(str)


def prf(tp_p, n_p, tp_g, n_g):
    P = tp_p / n_p if n_p else float("nan"); R = tp_g / n_g if n_g else float("nan")
    F = 2 * P * R / (P + R) if n_p and n_g and P + R else float("nan")
    return P, R, F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("alllabels"); ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]; al = [r for r in load_csv(a.alllabels) if r["doc_id"] in docs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rows = []

    def add(task, slice_kind, slice_val, n, **m): rows.append({"task": task, "slice": slice_kind, "value": slice_val, "n": n, "low_n": n < MIN_N, **{k: ("" if v != v else round(v, 3)) for k, v in m.items()}})

    # classification (out-of-fold)
    T = [doc_text(docs[r["doc_id"]]) for r in ev]
    pred = {f: oof(T, np.array([r[f] for r in ev])) for f in ("category", "autonomy_level", "interaction")}
    src = np.array([r["source_type"] for r in ev]); per = np.array([period(docs[r["doc_id"]]["publication_date"]) for r in ev])
    for f, p in pred.items():
        y = np.array([r[f] for r in ev])
        add(f, "all", "all", len(y), acc=accuracy_score(y, p), macro_f1=f1_score(y, p, average="macro", zero_division=0))
        for kind, arr in (("source_type", src), ("period", per)):
            for v in sorted(set(arr)):
                m = arr == v
                add(f, kind, v, int(m.sum()), acc=accuracy_score(y[m], p[m]), macro_f1=f1_score(y[m], p[m], average="macro", zero_division=0))
    Ta = [doc_text(docs[r["doc_id"]]) for r in al]; ya = np.array([r["has_event"] == "yes" for r in al]).astype(str)
    pa = oof(Ta, ya); sa = np.array([r["source_type"] for r in al]); pera = np.array([period(docs[r["doc_id"]]["publication_date"]) for r in al])
    add("has_event(yes/no)", "all", "all", len(ya), acc=accuracy_score(ya, pa), f1_yes=f1_score(ya, pa, pos_label="True", zero_division=0))
    for kind, arr in (("source_type", sa), ("period", pera)):
        for v in sorted(set(arr)):
            m = arr == v
            add("has_event(yes/no)", kind, v, int(m.sum()), acc=accuracy_score(ya[m], pa[m]), f1_yes=f1_score(ya[m], pa[m], pos_label="True", zero_division=0))

    # extraction (rule extractors)
    for field, gi in (("organization", 0), ("model_agent", 1)):
        def slice_m(idx):
            tp_p = n_p = tp_g = n_g = 0
            for i in idx:
                r = ev[i]; d = docs[r["doc_id"]]
                preds = guess_organizations(d) if gi == 0 else guess_systems(d); gold = vals(r[field])
                n_p += len(preds); n_g += len(gold)
                tp_p += sum(any(match(p, g) for g in gold) for p in preds); tp_g += sum(any(match(p, g) for p in preds) for g in gold)
            return prf(tp_p, n_p, tp_g, n_g)
        P, R, F = slice_m(range(len(ev))); add(field, "all", "all", len(ev), P=P, R=R, F1=F)
        for kind, arr in (("source_type", src), ("period", per)):
            for v in sorted(set(arr)):
                idx = np.where(arr == v)[0]; P, R, F = slice_m(idx); add(field, kind, v, len(idx), P=P, R=R, F1=F)

    cols = ["task", "slice", "value", "n", "low_n", "acc", "macro_f1", "f1_yes", "P", "R", "F1"]
    with open(out / "d16_robustness.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows([{c: r.get(c, "") for c in cols} for r in rows])
    L = []
    for task in dict.fromkeys(r["task"] for r in rows):
        L.append(f"\n== {task} ==")
        for r in [r for r in rows if r["task"] == task]:
            met = "  ".join(f"{k}={r[k]}" for k in ("acc", "macro_f1", "f1_yes", "P", "R", "F1") if r.get(k) not in (None, ""))
            L.append(f"  {r['slice']:12} {str(r['value']):13} n={r['n']:4}{'*' if r['low_n'] else ' '} {met}")
    txt = "\n".join(L); print(txt); (out / "d16_robustness_report.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
