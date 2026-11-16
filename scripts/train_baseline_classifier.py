"""D11: baseline classifiers on the frozen gold.

    python scripts/train_baseline_classifier.py <corpus.jsonl> <gold_events_v1_1.csv> <gold_all_labels_v1.csv> [--out results/tables]

1. category (8 classes): train on classifier_split == train, test on holdout; plus 5-fold CV on all events.
   Baselines: majority class, and source_type only.
2. has_event (yes / no_not_significant / no_off_topic): 5-fold CV over all labelled documents (no split is given).
"""
from __future__ import annotations
import argparse, csv, sys, warnings
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import doc_text, load_corpus, load_csv, make_model

warnings.filterwarnings("ignore")


def cv_pred(model, X, y, k=5, seed=0):
    cnt = Counter(y)
    k = max(2, min(k, min(cnt.values())))
    return cross_val_predict(model, X, y, cv=StratifiedKFold(k, shuffle=True, random_state=seed)), k


def line(name, y, p):
    return f"{name:34} acc={accuracy_score(y, p):.3f}  macro-F1={f1_score(y, p, average='macro', zero_division=0):.3f}  n={len(y)}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("alllabels"); ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]
    al = [r for r in load_csv(a.alllabels) if r["doc_id"] in docs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rep = []

    # ---- category ----
    X = [doc_text(docs[r["doc_id"]]) for r in ev]
    y = np.array([r["category"] for r in ev])
    tr = np.array([r["classifier_split"] == "train" for r in ev])
    Xtr, Xte = [x for x, t in zip(X, tr) if t], [x for x, t in zip(X, tr) if not t]
    ytr, yte = y[tr], y[~tr]
    rep.append(f"== category: train {len(ytr)} / holdout {len(yte)} ==")
    m = make_model().fit(Xtr, ytr)
    p = m.predict(Xte)
    rep.append(line("TF-IDF + LR (holdout)", yte, p))
    maj = DummyClassifier(strategy="most_frequent").fit(Xtr, ytr)
    rep.append(line("majority class (holdout)", yte, maj.predict(Xte)))
    st = make_pipeline(CountVectorizer(), LogisticRegression(max_iter=500)).fit([x.split()[0] for x in Xtr], ytr)
    rep.append(line("source_type only (holdout)", yte, st.predict([x.split()[0] for x in Xte])))
    pc, k = cv_pred(make_model(), X, y)
    rep.append(line(f"TF-IDF + LR ({k}-fold CV, all events)", y, pc))
    rep.append("\nholdout per-class report:\n" + classification_report(yte, p, zero_division=0))
    labels = sorted(set(y))
    rep.append("holdout confusion (rows=gold, cols=pred), order: " + ", ".join(labels))
    rep.append(str(confusion_matrix(yte, p, labels=labels)))
    with open(out / "d11_category_holdout_predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["doc_id", "title", "gold", "pred"])
        for r, g, q in zip([r for r, t in zip(ev, tr) if not t], yte, p):
            w.writerow([r["doc_id"], r["title"][:70], g, q])

    # ---- has_event ----
    Xa = [doc_text(docs[r["doc_id"]]) for r in al]
    ya = np.array([r["has_event"] for r in al])
    rep.append("\n== has_event (3 classes), CV over all labelled docs ==")
    rep.append("class counts: " + str(dict(Counter(ya))))
    pa, k = cv_pred(make_model(), Xa, ya)
    rep.append(line(f"TF-IDF + LR ({k}-fold CV)", ya, pa))
    rep.append(line("majority class", ya, np.array([Counter(ya).most_common(1)[0][0]] * len(ya))))
    rep.append(classification_report(ya, pa, zero_division=0))
    yb = np.array(["yes" if v == "yes" else "no" for v in ya])
    pb, _ = cv_pred(make_model(), Xa, yb)
    rep.append(line("yes vs no (binary, CV)", yb, pb))
    text = "\n".join(rep)
    (out / "d11_baseline_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
