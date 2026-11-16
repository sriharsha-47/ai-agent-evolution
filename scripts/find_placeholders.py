"""Flag placeholder / copy-paste / possibly out-of-scope rows in the gold events (owner: HA).

Usage (repo root):
  python scripts/find_placeholders.py
Writes results/tables/placeholder_rows.csv (send each annotator their own rows) and prints a summary per annotator.
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GOLD = ROOT / "data" / "annotations" / "gold_events_v1.csv"
CORPUS = ROOT / "data" / "interim" / "corpus_v1_clean.jsonl"
OUT = ROOT / "results" / "tables" / "placeholder_rows.csv"

PLACEHOLDER_MODEL = {"model/agent", "model_agent", "tbd", "todo", "n/a"}
NOT_AN_ORG = {"arxiv", "github"}
LM_TERMS = re.compile(r"\bLLMs?\b|language model|GPT|agentic|foundation model|tool[- ]use|function call|Claude|"
                      r"Gemini|\bMCP\b|copilot|chatbot|transformer|\bagents?\b", re.I)
POLICY_TITLE = re.compile(r"\b(why|update on|our approach to|lessons|principles|commitments|responsible|"
                          r"mental health|cooperation on safety)\b", re.I)


def main():
    gold = pd.read_csv(GOLD, dtype=str, keep_default_na=False)
    text = {}
    if CORPUS.exists():
        for line in open(CORPUS, encoding="utf-8"):
            r = json.loads(line)
            text[r["doc_id"]] = r["text"][:1500]
    else:
        print(f"(corpus not found at {CORPUS}; skipping the scope check)")

    # an action sentence repeated in >= 80% of one annotator's rows is a template value
    repeated = {}
    for who, g in gold.groupby("annotator"):
        if len(g) >= 5:
            act, n = Counter(g["action"]).most_common(1)[0]
            if act and n / len(g) >= 0.8:
                repeated[who] = act

    rows = []
    for _, r in gold.iterrows():
        flags = []
        if r["model_agent"].strip().lower() in PLACEHOLDER_MODEL:
            flags.append("placeholder model_agent")
        if r["organization"].strip().lower() in NOT_AN_ORG:
            flags.append("organization is a website, not an organisation (leave blank for papers)")
        if repeated.get(r["annotator"]) == r["action"]:
            flags.append("action text copied in most of this annotator's rows")
        if text and not LM_TERMS.search(r["title"] + " " + text.get(r["doc_id"], "")):
            flags.append("no language-model/agent term found: check scope (no_off_topic?)")
        if POLICY_TITLE.search(r["title"]):
            flags.append("title looks like policy/opinion: check no_not_significant")
        if flags:
            rows.append({"annotator": r["annotator"], "event_id": r["event_id"], "annotation_id": r["annotation_id"],
                         "title": r["title"], "category": r["category"], "flags": " | ".join(flags)})
    out = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False, encoding="utf-8")
    total = gold["annotator"].value_counts()
    print("rows flagged / events per annotator:")
    flagged = out["annotator"].value_counts() if len(out) else pd.Series(dtype=int)
    for who in total.index:
        print(f"  {who:10} {int(flagged.get(who, 0)):>3} / {int(total[who])}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    sys.exit(main())
