"""Check that a monthly feature table was built from the CURRENT gold events (owner: HA).

Usage (repo root):
  python scripts/check_feature_table.py data/processed/features_monthly_gold.csv data/annotations/gold_events_v1_1.csv
Recomputes events per month and the dominant category (min events and tie-break from the configs) and lists
months where the feature table disagrees. Any mismatch means the table is stale: ask Saketh to rebuild it.
"""
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ainevo.evaluation.gold import load_yaml  # noqa: E402
from ainevo.forecasting.splits import load_split, months  # noqa: E402


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    feats = pd.read_csv(sys.argv[1]).set_index("month")
    gold = pd.read_csv(sys.argv[2], dtype=str, keep_default_na=False)
    cfg = load_split(ROOT / "configs" / "split.yaml")
    prec = load_yaml(ROOT / "configs" / "taxonomy.yaml")["category_precedence"]
    min_n = cfg["target"]["min_events_per_month"]
    gold["m"] = gold["date"].str[:7]
    rows = []
    for m in months(cfg["series_start"], cfg["cutoff_month"]):
        sub = gold[gold["m"] == m]
        c = Counter(sub["category"])
        lab = None
        if len(sub) >= min_n:
            top = max(c.values())
            lab = [x for x in prec if c.get(x, 0) == top][0]
        fl = feats["dominant_category"].get(m)
        fl = fl if isinstance(fl, str) else None
        fn = int(feats["n_events"].get(m, 0))
        if fn != len(sub) or fl != lab:
            rows.append({"month": m, "n_events_table": fn, "n_events_gold": len(sub),
                         "label_table": fl, "label_gold": lab})
    if rows:
        print(f"STALE: {len(rows)} month(s) differ from {Path(sys.argv[2]).name}:")
        print(pd.DataFrame(rows).to_string(index=False))
        print("\nAsk Saketh to rebuild the feature table from the current gold file before you run baselines.")
    else:
        print("OK: the feature table matches the gold file (counts and monthly labels).")


if __name__ == "__main__":
    main()
