"""Run NER + date extraction on a JSONL corpus sample and write tidy CSVs.
Usage: python scripts/run_extraction_sample.py <in.jsonl> <out_prefix>
"""
import csv, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.extraction.ner import load_nlp, extract_entities
from ainevo.extraction.temporal import extract_dates, best_event_date

src, prefix = sys.argv[1], sys.argv[2]
nlp = load_nlp()
ent_rows, date_rows, doc_rows = [], [], []
for line in open(src, encoding="utf-8"):
    d = json.loads(line)
    text = d["title"] + ". " + d["text"]
    ents = extract_entities(d["text"], nlp, title=d["title"])
    dates = extract_dates(text, d.get("publication_date"), nlp)
    best, how = best_event_date(dates, d.get("publication_date"))
    for e in ents:
        ent_rows.append({"doc_id": d["doc_id"], **e})
    for x in dates:
        date_rows.append({"doc_id": d["doc_id"], "pub": d.get("publication_date"), **x})
    doc_rows.append({"doc_id": d["doc_id"], "pub": d.get("publication_date"), "n_entities": len(ents),
                     "n_dates": len(dates), "event_date": best, "event_date_how": how})
for name, rows in (("entities", ent_rows), ("dates", date_rows), ("docs", doc_rows)):
    if not rows: continue
    with open(f"{prefix}_{name}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print(len(doc_rows), "docs;", len(ent_rows), "entities;", len(date_rows), "dates")
