"""Draw a FRESH random sample (not in gold) for choosing the has_event threshold, as an Excel sheet to label.

    python scripts/make_threshold_sample.py data/interim/corpus_v1_clean.jsonl data/processed/gold_all_labels_v1.csv [--per-source 50] [--out data/annotations/threshold_sample.xlsx]

Unlike the gold pool this sample is NOT filtered toward events: only the corpus window (2017-01-01..2026-09-30) and a
150-character minimum are applied, so it looks like what the extractor sees. Label has_event only:
yes / no_not_significant / no_off_topic (guideline v1, section 3). Seed 7.
"""
from __future__ import annotations
import argparse, random, re, sys
from pathlib import Path
import openpyxl
from openpyxl.worksheet.datavalidation import DataValidation

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import load_corpus, load_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("alllabels"); ap.add_argument("--per-source", type=int, default=50)
    ap.add_argument("--out", default="data/annotations/threshold_sample.xlsx")
    a = ap.parse_args()
    docs = load_corpus(a.corpus)
    gold = {r["doc_id"] for r in load_csv(a.alllabels)}
    random.seed(7)
    wb = openpyxl.Workbook(); ws = wb.active; ws.title = "Sample"
    ws.append(["doc_id", "publication_date", "source", "source_type", "title", "url", "text_excerpt", "has_event", "notes"])
    for st in ("lab_blog", "paper", "release_notes"):
        pool = [d for k, d in docs.items() if k not in gold and d["source_type"] == st and len(d["text"]) >= 150
                and "2017-01-01" <= d["publication_date"] <= "2026-09-30"]
        for d in random.sample(pool, min(a.per_source, len(pool))):
            ws.append([d["doc_id"], d["publication_date"], d["source"], st, d["title"], d.get("url"),
                       re.sub(r"\s+", " ", d["text"])[:700], None, None])
    dv = DataValidation(type="list", formula1='"yes,no_not_significant,no_off_topic,no_unclear_text"', allow_blank=True)
    ws.add_data_validation(dv); dv.add(f"H2:H{ws.max_row}")
    for col, w in zip("ABCDEFGHI", (14, 12, 18, 13, 45, 30, 90, 20, 30)):
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "A2"
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    wb.save(a.out)
    print(a.out, ws.max_row - 1, "rows")


if __name__ == "__main__":
    main()
