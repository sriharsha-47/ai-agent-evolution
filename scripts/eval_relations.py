"""D9: evaluate relation extraction on the gold events.

    python scripts/eval_relations.py <corpus.jsonl> <gold_events_v1_1.csv> [--out results/tables]

Reports (a) organisation-system pair P/R/F1 (lenient match), (b) capability accuracy/macro-F1 for the keyword rules and a
TF-IDF + LR classifier (train split -> holdout, plus 5-fold CV), (c) action phrase unigram-F1 vs the gold action text.
"""
from __future__ import annotations
import argparse, csv, re, sys, warnings
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import load_corpus, load_csv, make_model
from ainevo.extraction.relations import extract_relation, rule_capability

warnings.filterwarnings("ignore")


def norm(s): return re.sub(r"[^a-z0-9]", "", s.lower())


def match(a, b):
    x, y = norm(a), norm(b)
    return len(x) >= 2 and len(y) >= 2 and (x in y or y in x)


def split_vals(v):
    return [p.strip() for p in re.split(r"[;,]", v or "") if p.strip()]


def toks(s): return re.findall(r"[a-z0-9]+", (s or "").lower())


def uni_f1(pred, gold):
    p, g = toks(pred), toks(gold)
    if not p or not g: return 0.0
    common = sum(min(p.count(w), g.count(w)) for w in set(p))
    if not common: return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    rels = [extract_relation(docs[r["doc_id"]]) for r in ev]
    lines = [f"events: {len(ev)}"]

    # (a) organisation - system pairs (papers have no organisation in gold, so their pairs are empty by design)
    tp = npred = ngold = 0
    for r, p in zip(ev, rels):
        gold = [(o, s) for o in split_vals(r["organization"]) for s in split_vals(r["model_agent"])[:1]]
        pred = [(o, s) for o in p["organization"] for s in p["system"][:1]]
        npred += len(pred); ngold += len(gold)
        tp += sum(any(match(po, go) and match(ps, gs) for go, gs in gold) for po, ps in pred)
    P, R = tp / npred if npred else 0, tp / ngold if ngold else 0
    lines.append(f"org-system pairs (first system only): gold={ngold} pred={npred} tp={tp}  P={P:.2f} R={R:.2f} F1={2*P*R/(P+R) if P+R else 0:.2f}")

    # (b) capability
    y = np.array([r["capability"] for r in ev]); train = np.array([r["classifier_split"] == "train" for r in ev])
    texts = [docs[r["doc_id"]]["title"] + ". " + docs[r["doc_id"]]["text"][:3000] for r in ev]
    rule_pred = np.array([rule_capability(docs[r["doc_id"]]) for r in ev])
    maj = max(set(y[train]), key=list(y[train]).count)
    lines.append("\ncapability (13 classes)             holdout acc / macro-F1   all-events acc / macro-F1")
    f = lambda yy, pp: f"{accuracy_score(yy, pp):.3f} / {f1_score(yy, pp, average='macro', zero_division=0):.3f}"
    lines.append(f"majority class ({maj})".ljust(36) + f"{f(y[~train], [maj]*int((~train).sum()))}")
    lines.append("keyword rules".ljust(36) + f"{f(y[~train], rule_pred[~train])}            {f(y, rule_pred)}")
    m = make_model().fit([t for t, k in zip(texts, train) if k], y[train])
    lr_hold = m.predict([t for t, k in zip(texts, train) if not k])
    pr = np.empty(len(y), dtype=object)
    for tri, tei in StratifiedKFold(5, shuffle=True, random_state=0).split(texts, np.where(np.isin(y, [c for c in set(y) if list(y).count(c) >= 5]), y, "other")):
        mm = make_model().fit([texts[i] for i in tri], y[tri]); pr[tei] = mm.predict([texts[i] for i in tei])
    lines.append("TF-IDF + LR".ljust(36) + f"{f(y[~train], lr_hold)}            {f(y, pr.astype(str))} (5-fold CV)")

    # (c) action
    ex = [uni_f1(p["action"], r["action"]) for p, r in zip(rels, ev)]
    ti = [uni_f1(docs[r["doc_id"]]["title"], r["action"]) for r in ev]
    fs = [uni_f1(" ".join(docs[r["doc_id"]]["text"].split()[:22]), r["action"]) for r in ev]
    lines.append(f"\naction phrase unigram-F1 vs gold:  extractive trigger clause={np.mean(ex):.3f}   title={np.mean(ti):.3f}   first 22 words={np.mean(fs):.3f}")
    for st in ("lab_blog", "paper", "release_notes"):
        ix = [i for i, r in enumerate(ev) if r["source_type"] == st]
        lines.append(f"   {st:14} n={len(ix):3} extractive={np.mean([ex[i] for i in ix]):.3f} title={np.mean([ti[i] for i in ix]):.3f}")
    text = "\n".join(lines); print(text)
    (out / "d9_relation_report.txt").write_text(text + "\n", encoding="utf-8")
    with open(out / "d9_relations_gold_events.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["doc_id", "title", "pred_org", "pred_system", "pred_capability", "gold_capability", "pred_action", "gold_action"])
        for r, p, rp in zip(ev, rels, rule_pred):
            w.writerow([r["doc_id"], r["title"][:60], "; ".join(p["organization"]), "; ".join(p["system"]), rp, r["capability"], p["action"], r["action"]])


if __name__ == "__main__":
    main()
