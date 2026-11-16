"""Write data/processed/forecast_latest.json for the Streamlit Forecast page (owner: HA).

Order of work (repo root):
  1. python scripts/run_baselines.py                       # validation block, repeat as needed -> pick the model
  2. python scripts/run_baselines.py --block test --final  # the ONE final test run
  3. python scripts/export_forecast.py                     # picks the best ML model on validation macro-F1
     (or: --model random_forest)
The forecast is for the month after the cutoff (cutoff 2026-09 -> forecast_month 2026-10). The model for that forecast is
refitted on all scored months. metrics come from the single test run; history is the validation + test predictions.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from ainevo.forecasting import baselines, splits  # noqa: E402

ML = ["logreg", "random_forest", "gradient_boosting"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / "features_monthly_gold.csv"))
    ap.add_argument("--events", default=str(ROOT / "data" / "annotations" / "gold_events_v1_1.csv"))
    ap.add_argument("--model", default=None, choices=ML)
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "forecast_latest.json"))
    ap.add_argument("--shrink", type=float, default=0.0,
                    help="0..1: blend the probabilities with the training class frequencies (0.5 = halfway). "
                         "Tame over-confident models; say so in the report.")
    a = ap.parse_args()

    cfg = splits.load_split(ROOT / "configs" / "split.yaml")
    ds = baselines.build_dataset(pd.read_csv(a.features), cfg)
    met = ROOT / "results" / "metrics"

    val_path, test_path = met / "forecast_metrics_validation.csv", met / "forecast_metrics_test.csv"
    if not test_path.exists():
        raise SystemExit("No test metrics yet. Run `python scripts/run_baselines.py --block test --final` once first.")
    if a.model is None:
        if not val_path.exists():
            raise SystemExit("No validation metrics: run `python scripts/run_baselines.py` first, or pass --model.")
        val = pd.read_csv(val_path)
        val = val[val["model"].isin(ML)].sort_values(["macro_f1", "accuracy"], ascending=False)
        a.model = val.iloc[0]["model"]
    print("model:", a.model)

    train = ds[ds["scored"]]
    model = baselines._make(a.model).fit(train[baselines.COLS], train["target"])
    last = ds[ds["month"] == cfg["cutoff_month"]]
    if last.empty:
        raise SystemExit(f"Feature table has no row for the cutoff month {cfg['cutoff_month']}.")
    row = last.iloc[0]
    proba = model.predict_proba(last[baselines.COLS])[0]
    classes = list(model.classes_)
    if a.shrink > 0:
        prior = train["target"].value_counts(normalize=True)
        prior = np.array([float(prior.get(c, 0.0)) for c in classes])
        proba = (1 - a.shrink) * proba + a.shrink * prior
    predicted = classes[int(np.argmax(proba))]

    tm = pd.read_csv(test_path)
    tm = tm[tm["model"] == a.model].iloc[0]

    hist = []
    for blk in ("validation", "test"):
        p = ROOT / "results" / "tables" / f"forecast_predictions_{blk}.csv"
        if p.exists():
            d = pd.read_csv(p)
            d = d[d["model"] == a.model]
            hist += [{"month": r.target_month, "actual": r.y_true, "predicted": r.y_pred} for r in d.itertuples()]
    hist.sort(key=lambda x: x["month"])

    clf = model.named_steps["clf"]
    names = [n.split("__", 1)[-1] for n in model.named_steps["pre"].get_feature_names_out()]
    imp = clf.feature_importances_ if hasattr(clf, "feature_importances_") else np.abs(clf.coef_).mean(axis=0)
    top = sorted(zip(names, imp), key=lambda x: -x[1])[:8]

    ev = pd.read_csv(a.events, dtype=str, keep_default_na=False)
    ev = ev[ev["date"].str[:7] == cfg["cutoff_month"]].copy()
    ev["match"] = (ev["category"] == predicted).astype(int)
    support = list(ev.sort_values(["match", "date"], ascending=[False, False])["event_id"].head(5))

    out = {
        "forecast_month": splits.next_month(cfg["cutoff_month"]),
        "model": a.model,
        "predicted_category": predicted,
        "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
        "baselines": {
            "persistence": row["dominant_category"] if pd.notna(row["dominant_category"]) else baselines._most_frequent(train),
            "most_frequent": baselines._most_frequent(train),
            "moving_average": baselines.trend_rule(train, row, ds),
        },
        "metrics": {"accuracy": round(float(tm["accuracy"]), 4), "macro_f1": round(float(tm["macro_f1"]), 4),
                    "test_months": int(tm["n"])},
        "history": hist,
        "top_features": [{"feature": f, "importance": round(float(i), 4)} for f, i in top],
        "supporting_event_ids": support,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"wrote {a.out}")
    print(json.dumps({k: out[k] for k in ("forecast_month", "predicted_category", "baselines", "metrics")}, indent=2))


if __name__ == "__main__":
    main()
