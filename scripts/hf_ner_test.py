"""Hugging Face NER test (run on YOUR laptop - the cloud workspace cannot reach huggingface.co).

Setup (Windows, Python 3.12 venv, RTX 4050):
    pip install transformers
    pip install torch --index-url https://download.pytorch.org/whl/cu121     # CUDA build; plain `pip install torch` = CPU
Usage (from repo root):
    python scripts/hf_ner_test.py data/interim/sample_20.jsonl results/tables/sample20_hf_entities.csv
"""
import csv, json, sys
import torch
from transformers import pipeline

src, out = sys.argv[1], sys.argv[2]
ner = pipeline("token-classification", model="dslim/bert-base-NER", aggregation_strategy="max",
               device=0 if torch.cuda.is_available() else -1)
rows = []
for line in open(src, encoding="utf-8"):
    d = json.loads(line)
    text, step = d["text"], 1200           # BERT sees at most 512 tokens: process long texts in chunks
    for off in range(0, len(text), step):
        chunk = text[off:off + step]
        for e in ner(chunk):
            rows.append({"doc_id": d["doc_id"], "text": chunk[e["start"]:e["end"]], "label": e["entity_group"],
                         "score": round(float(e["score"]), 3), "start": off + e["start"], "end": off + e["end"]})
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["doc_id", "text", "label", "score", "start", "end"])
    w.writeheader(); w.writerows(rows)
print(len(rows), "entities ->", out, "| device:", "cuda" if torch.cuda.is_available() else "cpu")
