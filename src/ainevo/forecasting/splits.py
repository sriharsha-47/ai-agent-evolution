"""Forecasting split + target definition (owner: HA).

Target: the dominant AI canon-event category in month t+1 (ties -> taxonomy category_precedence).
A row for month t holds the label of month t+1; its block (train / validation / test) is the block of the
MONTH BEING PREDICTED (t+1). Features for the row may only use events dated <= t (enforced later in features/).
Months with fewer than target.min_events_per_month events get no label and are not scored.
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
import yaml


def load_split(path="configs/split.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def months(start: str, end: str) -> list[str]:
    return [str(p) for p in pd.period_range(start, end, freq="M")]


def next_month(m: str) -> str:
    return str(pd.Period(m, freq="M") + 1)


def block_of(month, cfg: dict):
    """'train' / 'validation' / 'test' or None if the month is outside every block."""
    if not isinstance(month, str):
        return None
    for name, (a, b) in cfg["blocks"].items():
        if a <= month <= b:
            return name
    return None


def validate_split(cfg: dict) -> list[str]:
    """Return a list of problems (empty list = split config is consistent)."""
    problems = []
    b = cfg["blocks"]
    if b["train"][0] != cfg["series_start"]:
        problems.append("train must start at series_start")
    if next_month(b["train"][1]) != b["validation"][0]:
        problems.append("validation must start the month after train ends")
    if next_month(b["validation"][1]) != b["test"][0]:
        problems.append("test must start the month after validation ends")
    if b["test"][1] != cfg["cutoff_month"]:
        problems.append("test must end at cutoff_month")
    for name, (a, z) in b.items():
        if a > z:
            problems.append(f"block {name} starts after it ends")
    return problems


def dominant_category(counts: dict, precedence: list[str]):
    """Category with the most events; ties go to the one listed first in `precedence`. None if no events."""
    if not counts or sum(counts.values()) == 0:
        return None
    unknown = set(counts) - set(precedence)
    if unknown:
        raise ValueError(f"categories not in precedence list: {sorted(unknown)}")
    top = max(counts.values())
    return [c for c in precedence if counts.get(c, 0) == top][0]


def monthly_labels(events: pd.DataFrame, cfg: dict, precedence: list[str]) -> pd.DataFrame:
    """events needs columns `date` (YYYY-MM-DD or YYYY-MM) and `category` (one row per normalised event).

    Returns one row per month of the forecasting series with:
      month, n_events, dominant_category (label of this month),
      target_month (= next month), target_next (label to predict), block (block of target_month),
      scored (True if target_next exists and falls in a block).
    """
    ev = events.copy()
    ev["month"] = ev["date"].astype(str).str[:7]
    ev = ev[(ev["month"] >= cfg["series_start"]) & (ev["month"] <= cfg["cutoff_month"])]
    min_n = cfg["target"]["min_events_per_month"]

    rows = []
    for m in months(cfg["series_start"], cfg["cutoff_month"]):
        sub = ev[ev["month"] == m]
        n = len(sub)
        label = dominant_category(Counter(sub["category"]), precedence) if n >= min_n else None
        rows.append({"month": m, "n_events": n, "dominant_category": label})

    df = pd.DataFrame(rows)
    df["target_month"] = df["month"].map(next_month)
    df["target_next"] = df["dominant_category"].shift(-1)
    # the last row predicts a month after the cutoff -> never scored
    df.loc[df.index[-1], ["target_month", "target_next"]] = [None, None]
    df["block"] = df["target_month"].map(lambda m: block_of(m, cfg))
    df["scored"] = df["target_next"].notna() & df["block"].notna()
    return df


def describe_target(df: pd.DataFrame) -> pd.DataFrame:
    """Per block: scored months and the share of the most common label (= accuracy of the majority baseline)."""
    rows = []
    for blk in ("train", "validation", "test"):
        sub = df[(df["block"] == blk) & df["scored"]]
        if sub.empty:
            rows.append({"block": blk, "scored_months": 0, "majority_label": None, "majority_share": None})
            continue
        vc = sub["target_next"].value_counts()
        rows.append({"block": blk, "scored_months": len(sub), "majority_label": vc.index[0],
                     "majority_share": round(vc.iloc[0] / len(sub), 2),
                     "label_counts": vc.to_dict()})
    return pd.DataFrame(rows)
