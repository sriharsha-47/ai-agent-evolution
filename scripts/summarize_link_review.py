"""Summarise the manual same-event review of linked pairs (owner: HA, evaluates Saketh's linking).

Usage (repo root):
  python scripts/summarize_link_review.py data/extracted_link_review100_HA_draft.csv
Reads the column 'same_event? (Y/N)'. Prints pair-level precision, group-level results and the worst groups,
and writes results/metrics/link_review_summary.csv and results/tables/link_false_merges.csv.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COL = "same_event? (Y/N)"


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    r = pd.read_csv(sys.argv[1], dtype=str, keep_default_na=False)
    r[COL] = r[COL].str.strip().str.upper()
    bad = r[~r[COL].isin(["Y", "N"])]
    if len(bad):
        raise SystemExit(f"{len(bad)} rows still have no Y/N in '{COL}' (first: row {bad.index[0] + 2}).")
    r["y"] = (r[COL] == "Y").astype(int)
    g = r.groupby("event_group").agg(pairs=("y", "size"), same_event=("y", "sum"))
    g["share_correct"] = (g["same_event"] / g["pairs"]).round(2)
    g["verdict"] = pd.cut(g["share_correct"], [-0.01, 0.0, 0.99, 1.0], labels=["wrong merge", "mixed", "clean"])

    summary = pd.DataFrame([{
        "pairs_reviewed": len(r), "pairs_same_event": int(r["y"].sum()),
        "pair_precision": round(r["y"].mean(), 3),
        "groups": len(g), "groups_clean": int((g["verdict"] == "clean").sum()),
        "groups_mixed": int((g["verdict"] == "mixed").sum()), "groups_wrong": int((g["verdict"] == "wrong merge").sum()),
        "group_precision_clean_only": round((g["verdict"] == "clean").mean(), 3),
    }])
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    summary.to_csv(ROOT / "results" / "metrics" / "link_review_summary.csv", index=False)
    r[r[COL] == "N"].drop(columns="y").to_csv(ROOT / "results" / "tables" / "link_false_merges.csv", index=False)
    print(summary.T.to_string(header=False))
    print("\nworst groups:")
    print(g.sort_values("share_correct").head(10).to_string())
    print("\nwrote results/metrics/link_review_summary.csv and results/tables/link_false_merges.csv")
    print("NOTE: the number depends on the rule. Strict rule used: same release/announcement = Y; customer stories,")
    print("different features or different versions on the same day = N. State the rule in the report.")


if __name__ == "__main__":
    main()
