"""D11b: compare feature sets for the category classifier, same split and CV as the D11 baseline.

    python scripts/train_improved_classifier.py <corpus.jsonl> <gold_events_v1_1.csv> [--emb data/processed/embeddings_bge_small.npz]

Variants: tfidf | feat | tfidf+feat | (with --emb) emb | emb+feat | emb+tfidf+feat.
Reports holdout accuracy/macro-F1 and 5-fold CV (3 seeds averaged) on all events.
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import doc_text, load_corpus, load_csv
from ainevo.classification.features import feature_vector

warnings.filterwarnings("ignore")


def build(parts, tr_idx, te_idx, texts, F, E, C, w_feat, w_emb):
    mats_tr, mats_te = [], []
    if "tfidf" in parts:
        v = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, strip_accents="unicode")
        mats_tr.append(v.fit_transform([texts[i] for i in tr_idx])); mats_te.append(v.transform([texts[i] for i in te_idx]))
    if "feat" in parts:
        mats_tr.append(csr_matrix(F[tr_idx] * w_feat)); mats_te.append(csr_matrix(F[te_idx] * w_feat))
    if "emb" in parts:
        mats_tr.append(csr_matrix(E[tr_idx] * w_emb)); mats_te.append(csr_matrix(E[te_idx] * w_emb))
    return hstack(mats_tr).tocsr(), hstack(mats_te).tocsr()


def fit_pred(parts, y, tr_idx, te_idx, texts, F, E, C=10.0, w_feat=1.0, w_emb=1.0):
    Xtr, Xte = build(parts, tr_idx, te_idx, texts, F, E, C, w_feat, w_emb)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, y[tr_idx])
    return clf.predict(Xte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("--emb")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]
    texts = [doc_text(docs[r["doc_id"]]) for r in ev]
    y = np.array([r["category"] for r in ev])
    F = np.array([feature_vector(docs[r["doc_id"]]) for r in ev])
    train = np.array([r["classifier_split"] == "train" for r in ev])
    tr_idx, te_idx = np.where(train)[0], np.where(~train)[0]
    E = None
    variants = {"tfidf": ["tfidf"], "feat": ["feat"], "tfidf+feat": ["tfidf", "feat"]}
    if a.emb:
        z = np.load(a.emb, allow_pickle=True)
        pos = {d: i for i, d in enumerate(z["doc_ids"])}
        E = np.array([z["X"][pos[r["doc_id"]]] for r in ev])
        variants.update({"emb": ["emb"], "emb+feat": ["emb", "feat"], "emb+tfidf+feat": ["emb", "tfidf", "feat"]})
        print("embedding model:", str(z["model"]))
    print(f"train {len(tr_idx)} / holdout {len(te_idx)}   (accuracy / macro-F1)")
    print(f"{'variant':18} {'holdout acc':>11} {'holdout F1':>10} {'CV acc':>7} {'CV F1':>6}")
    for name, parts in variants.items():
        p = fit_pred(parts, y, tr_idx, te_idx, texts, F, E)
        ha, hf = accuracy_score(y[te_idx], p), f1_score(y[te_idx], p, average="macro", zero_division=0)
        accs, f1s = [], []
        for seed in (0, 1, 2):
            pr = np.empty(len(y), dtype=object)
            for tri, tei in StratifiedKFold(5, shuffle=True, random_state=seed).split(texts, y):
                pr[tei] = fit_pred(parts, y, tri, tei, texts, F, E)
            accs.append(accuracy_score(y, pr)); f1s.append(f1_score(y, pr, average="macro", zero_division=0))
        print(f"{name:18} {ha:11.3f} {hf:10.3f} {np.mean(accs):7.3f} {np.mean(f1s):6.3f}")


if __name__ == "__main__":
    main()
