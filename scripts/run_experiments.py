"""Exploratory model ideas on the VALIDATION (or train) block ONLY. This script never touches the test block.

  python scripts/run_experiments.py                 # validation block
Compares: strongly regularised logistic regression, shallow random forest, each with and without lag / rolling
features, against the baselines. Results go to results/metrics/experiments_<block>.csv.
Report these as EXPLORATORY: the headline result is still the single final test run that was already made.
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # noqa: E402

from ainevo.forecasting import baselines, splits  # noqa: E402

LAG_BASE = ["n_events", "entropy", "multi_agent_share", "incident_share", "high_autonomy_share"]
EXTRA = [c + s for c in LAG_BASE for s in ("_lag1", "_roll3")]


def add_lags(ds):
    d = ds.copy()
    for c in LAG_BASE:
        d[c + "_lag1"] = d[c].shift(1)                       # value of the previous month
        d[c + "_roll3"] = d[c].rolling(3, min_periods=1).mean()   # mean of the last 3 months, up to and including t
    return d


def make(kind, num_cols):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["last_label"]),
    ])
    clf = {
        "logreg_strong": LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"),
        "rf_shallow": RandomForestClassifier(n_estimators=300, max_depth=3, min_samples_leaf=2,
                                             random_state=baselines.SEED, class_weight="balanced_subsample"),
    }[kind]
    return Pipeline([("pre", pre), ("clf", clf)])


VARIANTS = {
    "logreg_strong": ("logreg_strong", baselines.NUM),
    "logreg_strong+lags": ("logreg_strong", baselines.NUM + EXTRA),
    "rf_shallow": ("rf_shallow", baselines.NUM),
    "rf_shallow+lags": ("rf_shallow", baselines.NUM + EXTRA),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / "features_monthly_gold.csv"))
    ap.add_argument("--block", default="validation", choices=["train", "validation"])
    a = ap.parse_args()

    cfg = splits.load_split(ROOT / "configs" / "split.yaml")
    ds = add_lags(baselines.build_dataset(pd.read_csv(a.features), cfg))
    ev = ds[(ds["block"] == a.block) & ds["scored"]]
    if ev.empty:
        raise SystemExit(f"No scored months in block {a.block}.")

    rows = []
    for idx, row in ev.iterrows():
        train = ds[ds["scored"] & (ds["target_month"] < row["target_month"])]
        preds = {}
        for name in ("most_frequent", "persistence", "moving_average_trend"):
            preds[name] = baselines.MODELS[name](train, row, ds)
        for name, (kind, cols) in VARIANTS.items():
            if len(train) < baselines.MIN_TRAIN_ROWS or train["target"].nunique() < 2:
                preds[name] = baselines._most_frequent(train)
            else:
                model = make(kind, cols).fit(train[cols + ["last_label"]], train["target"])
                preds[name] = model.predict(ds.loc[[idx], cols + ["last_label"]])[0]
        for name, p in preds.items():
            rows.append({"target_month": row["target_month"], "year": row["target_month"][:4], "model": name,
                         "y_true": row["target"], "y_pred": p})
    pred = pd.DataFrame(rows)
    overall, _, _ = baselines.score(pred)
    n = pred["target_month"].nunique()
    print(f"block={a.block}: {n} scored months (tiny sample: differences of one month are noise)")
    print(overall.round(3).to_string(index=False))
    (ROOT / "results" / "metrics").mkdir(parents=True, exist_ok=True)
    overall.to_csv(ROOT / "results" / "metrics" / f"experiments_{a.block}.csv", index=False)
    print(f"\nwrote results/metrics/experiments_{a.block}.csv")


if __name__ == "__main__":
    main()
