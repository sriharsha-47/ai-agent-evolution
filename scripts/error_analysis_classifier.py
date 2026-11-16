"""D13: error analysis of the category classifier (TF-IDF + LR baseline, out-of-fold over all gold events).

    python scripts/error_analysis_classifier.py data/interim/corpus_v1_clean.jsonl <gold_events_v1_1.csv> [--out results/tables] [--emb embeddings.npz]

Outputs: d13_confusion.csv, d13_confusion.png, d13_errors.csv (every wrong prediction with gold, pred, confidence, annotator,
source_type, title), d13_report.txt. With --emb the model is embeddings + LR (needs the npz from embed_docs.py).
Out-of-fold = each document is predicted by a model that never saw it (5-fold, seed 0), so all 217 events can be analysed.
"""
from __future__ import annotations
import argparse, csv, sys, warnings
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import doc_text, load_corpus, load_csv, make_model

warnings.filterwarnings("ignore")
ORDER = ["unexpected_behavior_incident", "multi_agent_coordination", "autonomous_action", "adoption_scale",
         "infrastructure_ecosystem", "agentic_development", "capability_breakthrough", "research_milestone"]  # taxonomy precedence


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("--out", default="results/tables"); ap.add_argument("--emb")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]
    y = np.array([r["category"] for r in ev])
    texts = [doc_text(docs[r["doc_id"]]) for r in ev]
    E = None
    if a.emb:
        z = np.load(a.emb, allow_pickle=True); pos = {d: i for i, d in enumerate(z["doc_ids"])}
        E = np.array([z["X"][pos[r["doc_id"]]] for r in ev])
    classes = sorted(set(y), key=ORDER.index)
    P = np.zeros((len(y), len(classes)))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(texts, y):
        if E is not None:
            m = LogisticRegression(C=10, max_iter=3000, class_weight="balanced").fit(E[tr], y[tr]); proba = m.predict_proba(E[te]); cl = list(m.classes_)
        else:
            m = make_model().fit([texts[i] for i in tr], y[tr]); proba = m.predict_proba([texts[i] for i in te]); cl = list(m.classes_)
        for j, c in enumerate(classes):
            P[te, j] = proba[:, cl.index(c)]
    pred = np.array(classes)[P.argmax(1)]
    top2 = np.array([y[i] in np.array(classes)[np.argsort(-P[i])[:2]] for i in range(len(y))])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    L = [f"model: {'embeddings + LR' if E is not None else 'TF-IDF + LR'}; out-of-fold over {len(y)} gold events",
         f"accuracy {accuracy_score(y, pred):.3f}  macro-F1 {f1_score(y, pred, average='macro', zero_division=0):.3f}  top-2 accuracy {top2.mean():.3f}"]
    cm = confusion_matrix(y, pred, labels=classes)
    with open(out / "d13_confusion.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["gold \\ pred"] + classes)
        for c, row in zip(classes, cm): w.writerow([c] + list(row))
    pr, rc, f1, sup = precision_recall_fscore_support(y, pred, labels=classes, zero_division=0)
    L.append("\nper class (taxonomy precedence order):  P     R     F1    n")
    for c, p_, r_, f_, s_ in zip(classes, pr, rc, f1, sup): L.append(f"  {c:30} {p_:.2f}  {r_:.2f}  {f_:.2f}  {s_}")
    pairs = Counter((g, p) for g, p in zip(y, pred) if g != p)
    L.append("\ntop confusions (gold -> predicted): count")
    for (g, p), n in pairs.most_common(8): L.append(f"  {g} -> {p}: {n}")
    hi = sum(1 for (g, p), n in pairs.items() if ORDER.index(p) < ORDER.index(g) for _ in range(n))
    lo = sum(n for (g, p), n in pairs.items() if ORDER.index(p) > ORDER.index(g))
    L.append(f"errors where predicted class has HIGHER precedence than gold: {hi}; LOWER precedence: {lo}  (of {sum(pairs.values())})")
    ann = np.array([r["annotator"] for r in ev]); st = np.array([r["source_type"] for r in ev]); conf = np.array([r["annotator_confidence"] for r in ev])
    for name, arr in (("annotator", ann), ("source_type", st), ("annotator_confidence", conf)):
        L.append(f"\naccuracy by {name}:")
        for v in sorted(set(arr)): m_ = arr == v; L.append(f"  {v:14} n={int(m_.sum()):3}  acc={accuracy_score(y[m_], pred[m_]):.3f}")
    wrong = np.where(pred != y)[0]
    conf_p = P.max(1)
    L.append(f"\nmean top probability: correct {conf_p[pred == y].mean():.2f}, wrong {conf_p[pred != y].mean():.2f}")
    with open(out / "d13_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["annotation_id", "doc_id", "annotator", "source_type", "annotator_confidence", "gold", "pred", "p_pred", "p_gold", "title"])
        for i in sorted(wrong, key=lambda i: -conf_p[i]):
            w.writerow([ev[i]["annotation_id"], ev[i]["doc_id"], ann[i], st[i], conf[i], y[i], pred[i], f"{conf_p[i]:.2f}", f"{P[i, classes.index(y[i])]:.2f}", ev[i]["title"][:80]])
    L.append("\nmost confident wrong predictions (likely label problems or rule ambiguity; check these first):")
    for i in sorted(wrong, key=lambda i: -conf_p[i])[:12]:
        L.append(f"  {ev[i]['annotation_id']:7} {ann[i]:9} gold={y[i]:26} pred={pred[i]:26} p={conf_p[i]:.2f} | {ev[i]['title'][:60]}")
    txt = "\n".join(L); print(txt); (out / "d13_report.txt").write_text(txt + "\n", encoding="utf-8")
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        norm = cm / cm.sum(1, keepdims=True).clip(1)
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        short = [c.replace("_", "\n") for c in classes]
        ax.set_xticks(range(len(classes)), short, fontsize=6.5); ax.set_yticks(range(len(classes)), short, fontsize=6.5)
        for i in range(len(classes)):
            for j in range(len(classes)): ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8, color="white" if norm[i, j] > .5 else "black")
        ax.set_xlabel("predicted"); ax.set_ylabel("gold"); ax.set_title("Category confusion (out-of-fold, rows normalised)")
        fig.tight_layout(); fig.savefig(out / "d13_confusion.png", dpi=160)
    except Exception as e:
        print("heatmap skipped:", e)


if __name__ == "__main__":
    main()
