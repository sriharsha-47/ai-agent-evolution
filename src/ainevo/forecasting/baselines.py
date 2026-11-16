"""Baselines + classical ML for 'dominant category next month' with rolling-origin evaluation (owner: HA).

Row for month t: features up to t, `last_label` = dominant category of month t, target = dominant category of t+1.
The block (train / validation / test) is the block of the month being predicted (t+1).
When predicting month m, models see only rows whose target month is < m (labels already known).
"""
from __future__ import annotations

from collections import Counter

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ainevo.forecasting.splits import block_of, next_month

NUM = ["n_events", "n_categories", "entropy", "multi_agent_share", "incident_share",
       "mean_autonomy", "high_autonomy_share", "n_orgs", "novelty"]
COLS = NUM + ["last_label"]
MIN_TRAIN_ROWS = 6
SEED = 42


def build_dataset(feat: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    f = feat.sort_values("month").reset_index(drop=True).copy()
    f["last_label"] = f["dominant_category"].fillna("none")
    f["target_month"] = f["month"].map(next_month)
    f["target"] = f["target_month"].map(dict(zip(f["month"], f["dominant_category"])))
    f["block"] = f["target_month"].map(lambda m: block_of(m, cfg))
    f["scored"] = f["target"].notna() & f["block"].notna()
    return f


# ---- baselines ----------------------------------------------------------------------------
def _most_frequent(train):
    return Counter(train["target"]).most_common(1)[0][0] if len(train) else None


def most_frequent(train, row, ds):
    return _most_frequent(train)


def persistence(train, row, ds):
    return row["dominant_category"] if pd.notna(row["dominant_category"]) else _most_frequent(train)


def trend_rule(train, row, ds, k=3):
    hist = ds[ds["month"] <= row["month"]]["dominant_category"].dropna().tail(k)
    return Counter(hist).most_common(1)[0][0] if len(hist) else _most_frequent(train)


# ---- classical ML -------------------------------------------------------------------------
def _make(name):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["last_label"]),
    ])
    clf = {
        "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced_subsample"),
        "gradient_boosting": GradientBoostingClassifier(random_state=SEED),
    }[name]
    return Pipeline([("pre", pre), ("clf", clf)])


def ml_model(name):
    def fn(train, row, ds):
        if len(train) < MIN_TRAIN_ROWS or train["target"].nunique() < 2:
            return _most_frequent(train)          # too little data: fall back to the majority label
        model = _make(name).fit(train[COLS], train["target"])
        return model.predict(ds.loc[[row.name], COLS])[0]
    return fn


MODELS = {
    "most_frequent": most_frequent,
    "persistence": persistence,
    "moving_average_trend": trend_rule,
    "logreg": ml_model("logreg"),
    "random_forest": ml_model("random_forest"),
    "gradient_boosting": ml_model("gradient_boosting"),
}


def rolling_eval(ds: pd.DataFrame, block: str) -> pd.DataFrame:
    ev = ds[(ds["block"] == block) & ds["scored"]]
    rows = []
    for idx, row in ev.iterrows():
        row = row.copy()
        row.name = idx
        train = ds[ds["scored"] & (ds["target_month"] < row["target_month"])]
        for name, fn in MODELS.items():
            rows.append({"target_month": row["target_month"], "year": row["target_month"][:4], "model": name,
                         "y_true": row["target"], "y_pred": fn(train, row, ds)})
    return pd.DataFrame(rows)


def score(pred: pd.DataFrame):
    """Returns (overall, by_year, per_class). Macro-F1 averages over classes that occur in y_true."""
    def one(g):
        labels = sorted(set(g["y_true"]))
        return pd.Series({"n": len(g), "accuracy": accuracy_score(g["y_true"], g["y_pred"]),
                          "macro_f1": f1_score(g["y_true"], g["y_pred"], labels=labels, average="macro", zero_division=0)})
    overall = pred.groupby("model")[["y_true", "y_pred"]].apply(one).reset_index()
    by_year = pred.groupby(["model", "year"])[["y_true", "y_pred"]].apply(one).reset_index()
    rows = []
    for model, g in pred.groupby("model"):
        labels = sorted(set(g["y_true"]))
        p, r, f, s = precision_recall_fscore_support(g["y_true"], g["y_pred"], labels=labels, zero_division=0)
        for lab, pi, ri, fi, si in zip(labels, p, r, f, s):
            rows.append({"model": model, "class": lab, "precision": pi, "recall": ri, "f1": fi, "support": si})
    return overall, by_year, pd.DataFrame(rows)
