"""D12: evaluate autonomy_level and interaction predictors on gold events.

    python scripts/eval_attributes.py data/interim/corpus_v1_clean.jsonl <gold_events_v1_1.csv> [--out results/tables]
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.attributes import rule_autonomy, rule_interaction
from ainevo.classification.baseline import doc_text, load_corpus, load_csv, make_model

warnings.filterwarnings("ignore")


def fmt(y, p): return f"{accuracy_score(y, p):.3f} / {f1_score(y, p, average='macro', zero_division=0):.3f}"


def run(name, field, ev, docs, rule_fn, out):
    y = np.array([r[field] for r in ev]); train = np.array([r["classifier_split"] == "train" for r in ev])
    texts = [doc_text(docs[r["doc_id"]]) for r in ev]
    maj = max(set(y[train]), key=list(y[train]).count)
    rule = np.array([rule_fn(docs[r["doc_id"]]) for r in ev])
    m = make_model().fit([t for t, k in zip(texts, train) if k], y[train])
    lr = m.predict([t for t, k in zip(texts, train) if not k])
    pr = np.empty(len(y), dtype=object)
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=0).split(texts, y):
        pr[tei] = make_model().fit([texts[i] for i in tri], y[tri]).predict([texts[i] for i in tei])
    L = [f"== {name} ({len(set(y))} classes), holdout n={int((~train).sum())}; acc / macro-F1 ==",
         f"majority ({maj})".ljust(26) + fmt(y[~train], [maj] * int((~train).sum())),
         "keyword rules".ljust(26) + f"holdout {fmt(y[~train], rule[~train])}    all {fmt(y, rule)}",
         "TF-IDF + LR".ljust(26) + f"holdout {fmt(y[~train], lr)}    5-fold CV {fmt(y, pr.astype(str))}"]
    # within-1 for the ordinal autonomy scale (NA excluded)
    if field == "autonomy_level":
        ok = [(a != "NA" and b != "NA") for a, b in zip(y, pr)]
        w1 = np.mean([abs(int(a) - int(b)) <= 1 for a, b, k in zip(y, pr, ok) if k])
        L.append(f"TF-IDF + LR CV: within one level (non-NA gold and pred) = {w1:.3f}")
    L.append("TF-IDF + LR CV per class:\n" + classification_report(y, pr.astype(str), zero_division=0))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]
    txt = run("autonomy_level", "autonomy_level", ev, docs, rule_autonomy, a.out) + "\n\n" + run("interaction", "interaction", ev, docs, rule_interaction, a.out)
    print(txt)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "d12_attribute_report.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
