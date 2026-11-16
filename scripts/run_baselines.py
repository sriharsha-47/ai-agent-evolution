"""Run baselines + LR/RF/GB with rolling-origin evaluation on a monthly feature table.

Usage (repo root):
  python scripts/run_baselines.py                       # validation block (safe to repeat)
  python scripts/run_baselines.py --block train         # sanity check on the training block
  python scripts/run_baselines.py --block test --final  # the ONE final test run (Week 3 only)
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from ainevo.forecasting import baselines, splits  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / "features_monthly_gold.csv"))
    ap.add_argument("--block", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--final", action="store_true", help="required to touch the test block")
    a = ap.parse_args()
    if a.block == "test" and not a.final:
        raise SystemExit("The test block is touched ONCE at the end. Re-run with --final when you are ready.")

    cfg = splits.load_split(ROOT / "configs" / "split.yaml")
    problems = splits.validate_split(cfg)
    if problems:
        raise SystemExit(f"split.yaml problems: {problems}")
    ds = baselines.build_dataset(pd.read_csv(a.features), cfg)
    scored = ds[ds["scored"]]
    print("scored rows per block:", scored["block"].value_counts().to_dict())

    pred = baselines.rolling_eval(ds, a.block)
    if pred.empty:
        raise SystemExit(f"No scored months in block {a.block}: not enough labelled months in this feature table.")
    overall, by_year, per_class = baselines.score(pred)

    n_eval = pred["target_month"].nunique()
    maj = pred.drop_duplicates("target_month")["y_true"].value_counts(normalize=True)
    print(f"\nblock={a.block}: {n_eval} scored months; most common label '{maj.index[0]}' = {maj.iloc[0]:.0%} of them")
    print(overall.round(3).to_string(index=False))

    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    pred.to_csv(ROOT / "results" / "tables" / f"forecast_predictions_{a.block}.csv", index=False)
    per_class.to_csv(ROOT / "results" / "tables" / f"forecast_per_class_{a.block}.csv", index=False)
    overall.to_csv(ROOT / "results" / "metrics" / f"forecast_metrics_{a.block}.csv", index=False)
    by_year.to_csv(ROOT / "results" / "metrics" / f"forecast_by_year_{a.block}.csv", index=False)
    print(f"\nwrote results/metrics/forecast_metrics_{a.block}.csv and 3 more files")
    print("NOTE: gold-sample features are a prototype; the numbers will be tiny-sample and unstable.")


if __name__ == "__main__":
    main()
