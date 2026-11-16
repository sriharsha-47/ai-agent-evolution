import pandas as pd

from ainevo.forecasting.splits import (block_of, dominant_category, monthly_labels,
                                       months, next_month, validate_split)

PREC = ["a", "b", "c"]
CFG = {
    "series_start": "2022-01",
    "cutoff_month": "2022-04",
    "blocks": {"train": ["2022-01", "2022-02"], "validation": ["2022-03", "2022-03"], "test": ["2022-04", "2022-04"]},
    "target": {"min_events_per_month": 2},
}


def test_months_and_next():
    assert months("2022-11", "2023-02") == ["2022-11", "2022-12", "2023-01", "2023-02"]
    assert next_month("2022-12") == "2023-01"


def test_dominant_category_tie_break_uses_precedence():
    assert dominant_category({"b": 2, "a": 2, "c": 1}, PREC) == "a"
    assert dominant_category({"c": 3, "b": 1}, PREC) == "c"
    assert dominant_category({}, PREC) is None


def test_block_of():
    assert block_of("2022-03", CFG) == "validation"
    assert block_of("2021-12", CFG) is None
    assert block_of(None, CFG) is None


def test_validate_split():
    assert validate_split(CFG) == []
    bad = {**CFG, "cutoff_month": "2022-06"}
    assert validate_split(bad) != []


def test_monthly_labels():
    ev = pd.DataFrame({
        "date": ["2022-01-05", "2022-01-09", "2022-02-01", "2022-02-02", "2022-02-03", "2022-03-01"],
        "category": ["b", "b", "a", "c", "a", "a"],
    })
    df = monthly_labels(ev, CFG, PREC).set_index("month")
    assert df.loc["2022-01", "dominant_category"] == "b"
    assert df.loc["2022-02", "dominant_category"] == "a"
    assert pd.isna(df.loc["2022-03", "dominant_category"])        # only 1 event < min 2 -> no label
    assert df.loc["2022-01", "target_next"] == "a"
    assert df.loc["2022-01", "block"] == "train"                  # block of the month being predicted (Feb)
    assert bool(df.loc["2022-01", "scored"]) is True
    assert bool(df.loc["2022-02", "scored"]) is False             # its target month (Mar) has no label
    assert bool(df.loc["2022-04", "scored"]) is False             # last row predicts after the cutoff
