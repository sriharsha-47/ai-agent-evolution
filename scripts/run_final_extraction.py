"""D12: final extraction run on the corpus: pipeline fields + predicted has_event, category, capability, autonomy, interaction.

    python scripts/run_final_extraction.py data/interim/corpus_v1_clean.jsonl <gold_events_v1_1.csv> <gold_all_labels_v1.csv> [--out data/processed]

Models (TF-IDF + logistic regression, trained on all gold) are fitted once; `in_gold` marks documents the models saw, whose
predictions are therefore in-sample. Attributes are filled only where has_event_pred == yes.
Re-run this (never edit the CSV by hand) after the corpus freeze or a gold change.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, sys, warnings
from collections import Counter
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import doc_text, load_corpus, load_csv, make_model
from ainevo.extraction.relations import extract_relation, rule_capability

warnings.filterwarnings("ignore")
COLS = ["doc_id", "source", "source_type", "title", "publication_date", "event_date", "p_event", "has_event_pred", "category_pred",
        "capability_pred", "autonomy_pred", "interaction_pred", "organizations", "systems", "action", "in_gold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("events"); ap.add_argument("alllabels"); ap.add_argument("--out", default="data/processed")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    ev = [r for r in load_csv(a.events) if r["doc_id"] in docs]
    al = [r for r in load_csv(a.alllabels) if r["doc_id"] in docs]
    gold_ids = {r["doc_id"] for r in al}
    T = lambda r: doc_text(docs[r["doc_id"]])
    m_event = make_model().fit([T(r) for r in al], [("yes" if r["has_event"] == "yes" else "no") for r in al])
    m_cat = make_model().fit([T(r) for r in ev], [r["category"] for r in ev])
    m_aut = make_model().fit([T(r) for r in ev], [r["autonomy_level"] for r in ev])
    m_int = make_model().fit([T(r) for r in ev], [r["interaction"] for r in ev])
    ids = list(docs)
    X = [doc_text(docs[i]) for i in ids]
    classes = list(m_event.classes_)
    p_yes = m_event.predict_proba(X)[:, classes.index("yes")]
    cat, aut, inter = m_cat.predict(X), m_aut.predict(X), m_int.predict(X)
    rows = []
    for k, i in enumerate(ids):
        d = docs[i]
        rel = extract_relation(d)
        yes = p_yes[k] >= 0.5
        rows.append({"doc_id": i, "source": d.get("source"), "source_type": d.get("source_type"), "title": d.get("title"),
                     "publication_date": d.get("publication_date"), "event_date": d.get("publication_date"),
                     "p_event": f"{p_yes[k]:.3f}", "has_event_pred": "yes" if yes else "no",
                     "category_pred": cat[k] if yes else "", "capability_pred": rule_capability(d) if yes else "",
                     "autonomy_pred": aut[k] if yes else "", "interaction_pred": inter[k] if yes else "",
                     "organizations": "; ".join(rel["organization"]), "systems": "; ".join(rel["system"]),
                     "action": rel["action"] if yes else "", "in_gold": i in gold_ids})
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with open(out / "extractions_final.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    h = hashlib.sha256(Path(a.corpus).read_bytes()).hexdigest()[:12]
    lines = [f"corpus: {a.corpus}  docs={len(rows)}  sha256[:12]={h}", f"gold: {len(ev)} events, {len(al)} labelled docs"]
    for st in ("lab_blog", "paper", "release_notes"):
        s = [r for r in rows if r["source_type"] == st]
        lines.append(f"{st:14} n={len(s):4} predicted events={sum(r['has_event_pred']=='yes' for r in s)} ({sum(r['has_event_pred']=='yes' for r in s)/len(s):.0%})")
    lines.append("category_pred: " + ", ".join(f"{k}={v}" for k, v in Counter(r["category_pred"] for r in rows if r["category_pred"]).most_common()))
    (out / "extraction_final_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
