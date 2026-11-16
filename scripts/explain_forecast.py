"""Week-3 explainability: feature importance + the events behind each forecast (owner: HA).

Usage (repo root), after run_baselines.py has written results/tables/forecast_predictions_<block>.csv:
  python scripts/explain_forecast.py --block validation --model random_forest \
      --timeline data/processed/timeline_gold_1.csv
Writes results/tables/forecast_evidence.csv (month, truth, prediction, supporting events with title + link)
and results/tables/feature_importance_<model>.csv (random_forest / gradient_boosting only).
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
    ap.add_argument("--block", default="validation", choices=["train", "validation", "test"])
    ap.add_argument("--model", default="random_forest")
    ap.add_argument("--features", default=str(ROOT / "data" / "processed" / "features_monthly_gold.csv"))
    ap.add_argument("--timeline", default=str(ROOT / "data" / "processed" / "timeline_gold_1.csv"))
    ap.add_argument("--top", type=int, default=3, help="supporting events per forecast month")
    a = ap.parse_args()

    pred_path = ROOT / "results" / "tables" / f"forecast_predictions_{a.block}.csv"
    pred = pd.read_csv(pred_path)
    pred = pred[pred["model"] == a.model]
    if pred.empty:
        raise SystemExit(f"No rows for model {a.model} in {pred_path.name}.")
    tl = pd.read_csv(a.timeline, dtype=str, keep_default_na=False)
    tl["m"] = tl["date"].str[:7]

    rows = []
    for _, p in pred.iterrows():
        month_t = str(pd.Period(p["target_month"], freq="M") - 1)          # the month whose events fed the forecast
        ev = tl[tl["m"] == month_t].copy()
        ev["match"] = (ev["category"] == p["y_pred"]).astype(int)          # events of the predicted category first
        ev = ev.sort_values(["match", "date"], ascending=[False, False]).head(a.top)
        rows.append({
            "target_month": p["target_month"], "y_true": p["y_true"], "y_pred": p["y_pred"],
            "correct": p["y_true"] == p["y_pred"], "features_from_month": month_t,
            "supporting_events": " || ".join(f"{r.date} [{r.category}] {r.title} ({r.source_url})" for r in ev.itertuples()),
        })
    out = pd.DataFrame(rows)
    (ROOT / "results" / "tables").mkdir(parents=True, exist_ok=True)
    out.to_csv(ROOT / "results" / "tables" / "forecast_evidence.csv", index=False, encoding="utf-8")
    print(out[["target_month", "y_true", "y_pred", "correct"]].to_string(index=False))
    print("\nwrote results/tables/forecast_evidence.csv")

    if a.model in ("random_forest", "gradient_boosting"):
        cfg = splits.load_split(ROOT / "configs" / "split.yaml")
        ds = baselines.build_dataset(pd.read_csv(a.features), cfg)
        train = ds[ds["scored"] & ds["block"].isin(["train", "validation"]) if a.block == "test" else ds["scored"] & (ds["block"] == "train")]
        if len(train) >= baselines.MIN_TRAIN_ROWS and train["target"].nunique() > 1:
            model = baselines._make(a.model).fit(train[baselines.COLS], train["target"])
            names = model.named_steps["pre"].get_feature_names_out()
            imp = pd.DataFrame({"feature": names, "importance": model.named_steps["clf"].feature_importances_})
            imp = imp.sort_values("importance", ascending=False)
            imp.to_csv(ROOT / "results" / "tables" / f"feature_importance_{a.model}.csv", index=False)
            print("\ntop features:\n" + imp.head(8).to_string(index=False))
        else:
            print("\nnot enough training rows for a feature-importance fit.")


if __name__ == "__main__":
    main()
