"""Embed documents with a sentence-transformer (run in Colab; the cloud workspace and some laptops block torch).

    pip install sentence-transformers
    python scripts/embed_docs.py <corpus.jsonl> <gold_all_labels_v1.csv> --out data/processed/embeddings_bge_small.npz [--model BAAI/bge-small-en-v1.5]

Only the labelled documents are embedded (395), so it takes about a minute on CPU. Output: npz with doc_ids and X.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.classification.baseline import doc_text, load_corpus, load_csv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus"); ap.add_argument("alllabels")
    ap.add_argument("--out", default="data/processed/embeddings_bge_small.npz")
    ap.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    docs = load_corpus(a.corpus)
    ids = [r["doc_id"] for r in load_csv(a.alllabels) if r["doc_id"] in docs]
    texts = [doc_text(docs[i])[:2000] for i in ids]  # model max length is 512 tokens; extra text is truncated anyway
    model = SentenceTransformer(a.model)
    X = model.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, doc_ids=np.array(ids), X=X, model=a.model)
    print(a.out, X.shape)


if __name__ == "__main__":
    main()
