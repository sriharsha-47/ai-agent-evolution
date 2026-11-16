"""Baseline NER evaluation (D5) + extraction error log (D6).

Usage:
    python scripts/eval_ner.py <corpus.jsonl> <annotation_SH.xlsx> [--spacy] [--out results/tables]

Compares predicted organisations / model-agent names with the gold `organization` and `model_agent` columns of the
rows with has_event == yes. Pilot rows (ids P01..P15) are reported separately from the whole sheet.
Variants:
    rules      gazetteer only over title + text (Task 1 rule layer)            (always)
    rules+spacy  Task 1 pipeline: gazetteer + spaCy ORG/PRODUCT               (only with --spacy)
    fixed      Task 5: source-derived org + title system name + top gazetteer  (always)
Matching is lenient: normalised (lower-case, alphanumerics only) containment in either direction, min length 2 (so "o1" matches "OpenAI o1").
No spaCy is needed unless --spacy is given.
"""
from __future__ import annotations
import csv, json, re, sys, argparse
from pathlib import Path
import openpyxl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.extraction.ner import (gazetteer_entities, guess_organizations, guess_systems, extract_entities, load_nlp)


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def split_gold(v) -> list[str]:
    if v is None or str(v).strip() in ("", "None"):
        return []
    return [p.strip() for p in str(v).split(";") if p.strip()]


def match(a: str, b: str) -> bool:
    x, y = norm(a), norm(b)
    return len(x) >= 2 and len(y) >= 2 and (x in y or y in x)


def load_gold(xlsx):
    wb = openpyxl.load_workbook(xlsx, data_only=True)
    rows = []
    for sh in ("Pilot", "Batch_SH"):
        ws = wb[sh]
        h = [c.value for c in ws[1]]
        for r in ws.iter_rows(min_row=2, values_only=True):
            r = dict(zip(h, r))
            if r.get("annotation_id") and str(r.get("has_event")).lower() == "yes":
                r["sheet"] = sh
                rows.append(r)
    return rows


def predict(variant, doc, nlp):
    text = doc["text"]
    full = (doc["title"] or "") + ". " + text
    if variant == "rules":
        return gazetteer_entities(full, "ORG"), gazetteer_entities(full, "MODEL_AGENT")
    if variant == "rules+spacy":
        ents = extract_entities(text, nlp, title=doc["title"])
        orgs, sys_ = [], []
        for e in ents:
            (sys_ if e["label"] in ("MODEL_AGENT", "PRODUCT") else orgs if e["label"] == "ORG" else []).append(e["text"])
        return list(dict.fromkeys(orgs)), list(dict.fromkeys(sys_))
    if variant == "fixed":
        return guess_organizations(doc), guess_systems(doc)
    raise ValueError(variant)


def score(rows, docs, variant, nlp, errors):
    tp_p = n_p = tp_g = n_g = 0
    docs_hit = docs_total = 0
    out = {}
    for field, gi in (("organization", 0), ("model_agent", 1)):
        tp_p = n_p = tp_g = n_g = docs_hit = docs_gold = abstain_ok = n_abst_gold = 0
        for r in rows:
            d = docs[r["doc_id"]]
            preds = predict(variant, d, nlp)[gi]
            gold = split_gold(r[field])
            n_p += len(preds); n_g += len(gold)
            tp_p += sum(any(match(p, g) for g in gold) for p in preds)
            hit_g = [any(match(p, g) for p in preds) for g in gold]
            tp_g += sum(hit_g)
            if gold:
                docs_gold += 1
                docs_hit += any(hit_g)
            else:
                n_abst_gold += 1
                abstain_ok += (len(preds) == 0)
            for g, ok in zip(gold, hit_g):
                if not ok:
                    errors.append(error_row(variant, field, r, d, g, preds))
            for p in preds:
                if gold and not any(match(p, g) for g in gold):
                    errors.append({"variant": variant, "field": field, "annotation_id": r["annotation_id"], "doc_id": r["doc_id"],
                                   "kind": "false_positive", "gold": "; ".join(gold), "pred": p, "reason": "predicted but not in gold",
                                   "title": d["title"][:60]})
                if not gold:
                    errors.append({"variant": variant, "field": field, "annotation_id": r["annotation_id"], "doc_id": r["doc_id"],
                                   "kind": "false_positive", "gold": "", "pred": p, "reason": "gold empty (paper/no org) but predicted",
                                   "title": d["title"][:60]})
        P = tp_p / n_p if n_p else 0.0
        R = tp_g / n_g if n_g else 0.0
        F = 2 * P * R / (P + R) if P + R else 0.0
        out[field] = dict(n_gold=n_g, n_pred=n_p, P=P, R=R, F1=F,
                          doc_hit=f"{docs_hit}/{docs_gold}", abstain_ok=f"{abstain_ok}/{n_abst_gold}")
    return out


def error_row(variant, field, r, d, g, preds):
    full = norm((d["title"] or "") + " " + d["text"])
    if norm(g) not in full:
        # gold not verbatim in text: descriptive label, expanded name, or annotator-added normalisation
        toks = [t for t in re.findall(r"[A-Za-z0-9\-\.]{3,}", g)]
        found = [t for t in toks if norm(t) in full]
        reason = "gold_descriptive_or_expanded" if found else "gold_not_in_text"
    elif not preds:
        reason = "abstained"
    elif any(set(re.findall(r"[a-z0-9]{3,}", p.lower())) & set(re.findall(r"[a-z0-9]{3,}", g.lower())) for p in preds):
        reason = "partial_token_overlap"
    else:
        reason = "wrong_or_missing_pick"
    return {"variant": variant, "field": field, "annotation_id": r["annotation_id"], "doc_id": r["doc_id"], "kind": "miss",
            "gold": g, "pred": "; ".join(preds), "reason": reason, "title": d["title"][:60]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("xlsx")
    ap.add_argument("--spacy", action="store_true"); ap.add_argument("--out", default="results/tables")
    a = ap.parse_args()
    docs = {}
    for l in open(a.corpus, encoding="utf-8"):
        d = json.loads(l); docs[d["doc_id"]] = d
    rows = [r for r in load_gold(a.xlsx) if r["doc_id"] in docs]
    nlp = load_nlp() if a.spacy else None
    variants = ["rules"] + (["rules+spacy"] if a.spacy else []) + ["fixed"]
    subsets = {"pilot": [r for r in rows if r["sheet"] == "Pilot"], "all": rows}
    Path(a.out).mkdir(parents=True, exist_ok=True)
    summary, errors = [], []
    for sname, srows in subsets.items():
        for v in variants:
            errs = []
            res = score(srows, docs, v, nlp, errs)
            if sname == "all":
                errors += errs
            for field, m in res.items():
                summary.append({"subset": sname, "n_events": len(srows), "variant": v, "field": field, **{k: (round(x, 3) if isinstance(x, float) else x) for k, x in m.items()}})
    with open(Path(a.out) / "task5_ner_scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0])); w.writeheader(); w.writerows(summary)
    with open(Path(a.out) / "task5_ner_errors.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(errors[0])); w.writeheader(); w.writerows(errors)
    print(f"{'subset':6} {'n':>3} {'variant':12} {'field':13} {'gold':>4} {'pred':>4}  P     R     F1    doc_hit  abstain_ok")
    for s in summary:
        print(f"{s['subset']:6} {s['n_events']:>3} {s['variant']:12} {s['field']:13} {s['n_gold']:>4} {s['n_pred']:>4}  {s['P']:.2f}  {s['R']:.2f}  {s['F1']:.2f}  {s['doc_hit']:>7}  {s['abstain_ok']}")
    from collections import Counter
    print("\nmiss reasons (fixed, all):")
    for (fld, k), n in sorted(Counter((e["field"], e["reason"]) for e in errors if e["variant"] == "fixed" and e["kind"] == "miss").items()):
        print(f"  {fld:13} {k:30} {n}")


if __name__ == "__main__":
    main()
