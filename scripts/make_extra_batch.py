"""Sample EXTRA documents (not yet labelled) to reach 200+ gold events, oversampling the rare categories.

Usage (repo root):
  python scripts/make_extra_batch.py            # 80 docs -> data/annotations/annotation_extra_HA.xlsx
  python scripts/make_extra_batch.py --n 100
Label sheet 'Batch_EXTRA' like any other batch, then run:
  python scripts/week1_gold.py validate-extra
  python scripts/week1_gold.py merge          # merge picks up annotation_extra_*.xlsx automatically
Note: this sample is deliberately biased toward incident/autonomy/adoption documents. Say so in the report.
"""
import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openpyxl import Workbook  # noqa: E402
from openpyxl.styles import Alignment, Font, PatternFill  # noqa: E402
from openpyxl.worksheet.datavalidation import DataValidation  # noqa: E402

from ainevo.evaluation import gold  # noqa: E402

ROOT = gold.ROOT
CUT = "2026-09-30"
HINT = {  # keyword -> weight (rare categories weigh most)
    "incident": (r"incident|vulnerab|jailbreak|unexpected|misalign|attack|exploit|misuse|rogue|emergent", 3),
    "auton": (r"autonomous|long-horizon|computer use|browser agent|operator|self-driving|without human", 3),
    "adopt": (r"deploy|adoption|million|enterprise|rollout|available to all|generally available", 2),
    "infra": (r"\bAPI\b|\bSDK\b|\bMCP\b|\bA2A\b|platform|infrastructure|plugin", 1),
    "multi": (r"multi-agent|multiagent|agent-to-agent|swarm|delegat|orchestrat", 1),
}
SHARE = {"paper": 0.5, "lab_blog": 0.35, "release_notes": 0.15}
LLM = re.compile(r"\bLLMs?\b|language model|GPT|agentic|foundation model|tool[- ]use|function call|Claude|Gemini|\bMCP\b|copilot|chatbot", re.I)


def keep(r):
    m = r.get("metadata", {})
    if len(r["text"]) < 150 or not ("2017-01-01" <= r["publication_date"] <= CUT):
        return False
    if r["source_type"] == "release_notes" and (m.get("low_signal_release") or len(r["text"]) < 300):
        return False
    if r["source_type"] == "paper" and r["publication_date"] >= "2023" and not m.get("llm_terms_found"):
        return False
    return True


def score(r):
    t = r["title"] + " " + r["text"][:1500]
    return sum(w for p, w in HINT.values() if re.search(p, t, re.I))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    ann = ROOT / "data" / "annotations"

    used = set()
    for w in gold.MEMBERS:
        used |= set(gold.read_pilot(w, ann)["doc_id"]) | set(gold.read_batch(w, ann)["doc_id"])
    rows = [json.loads(l) for l in open(ROOT / "data" / "interim" / "corpus_v1_clean.jsonl", encoding="utf-8")]
    cand = [r for r in rows if r["doc_id"] not in used and keep(r)]
    rng = random.Random(a.seed)
    rng.shuffle(cand)
    for r in cand:
        r["_s"] = score(r)
    cand.sort(key=lambda r: -r["_s"])          # stable: ties stay shuffled

    chosen = []
    for st, share in SHARE.items():
        pool = [r for r in cand if r["source_type"] == st and r["_s"] > 0]
        chosen += pool[: round(a.n * share)]
    ids = {r["doc_id"] for r in chosen}
    for r in cand:                              # top up if a source type ran short
        if len(chosen) >= a.n:
            break
        if r["doc_id"] not in ids:
            chosen.append(r)
            ids.add(r["doc_id"])
    chosen.sort(key=lambda r: r["publication_date"])

    wb = Workbook()
    rd = wb.active
    rd.title = "README"
    for line in ["EXTRA batch (HA). Label like any other batch; if has_event is not 'yes' leave the yellow columns empty.",
                 "These documents were sampled with a bias toward incident / autonomy / adoption keywords.",
                 "trigger = exact words from the text. Not sure: confidence=1 and add a note."]:
        rd.append([line])
    ws = wb.create_sheet("Batch_EXTRA")
    cols = gold.META + gold.FIELDS
    ws.append(cols)
    for i, r in enumerate(chosen, 1):
        ws.append([f"EX-{i:03d}", r["doc_id"], r["publication_date"], r["source"], r["source_type"], r["title"],
                   r["url"], re.sub(r"\s+", " ", r["text"])[:700]] + [""] * len(gold.FIELDS))
    tax = gold.load_yaml(ROOT / "configs" / "taxonomy.yaml")
    lists = {
        "has_event": tax["has_event_values"], "category": [c["id"] for c in tax["categories"]],
        "capability": tax["capability_tags"], "autonomy_level": ["0", "1", "2", "3", "4", "NA"],
        "interaction": tax["interaction"], "phase_hint": [str(i) for i in range(1, 8)], "confidence": ["1", "2", "3"],
    }
    head, fill = PatternFill("solid", fgColor="DDEBF7"), PatternFill("solid", fgColor="FFF2CC")
    for j, c in enumerate(cols, 1):
        cell = ws.cell(1, j)
        cell.font = Font(bold=True)
        cell.fill = head if c in gold.META else fill
        ws.column_dimensions[cell.column_letter].width = 70 if c == "text_excerpt" else (40 if c == "title" else 18)
    for row in ws.iter_rows(min_row=2):
        for cell in row:
            cell.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "G2"
    n = len(chosen) + 1
    for f, vals in lists.items():
        col = ws.cell(1, cols.index(f) + 1).column_letter
        dv = DataValidation(type="list", formula1='"' + ",".join(vals) + '"', allow_blank=True)
        ws.add_data_validation(dv)
        dv.add(f"{col}2:{col}{n}")
    out = ann / "annotation_extra_HA.xlsx"
    wb.save(out)
    counts = {st: sum(1 for r in chosen if r["source_type"] == st) for st in SHARE}
    print(f"wrote {out} with {len(chosen)} documents {counts}")


if __name__ == "__main__":
    main()
