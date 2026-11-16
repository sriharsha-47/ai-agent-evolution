import pandas as pd

from ainevo.evaluation.gold import FIELDS, META, _clean, assign_holdout, validate

TAX = {
    "categories": [
        {"id": "agentic_development", "default_phase": [3]},
        {"id": "research_milestone", "default_phase": [1, 2, 3, 4, 5, 6, 7]},
    ],
    "has_event_values": ["yes", "no_not_significant", "no_off_topic", "no_unclear_text"],
    "capability_tags": ["tool_use", "other"],
    "interaction": ["agent_tool", "human_model"],
}
GOOD = dict(has_event="yes", event_date="2024-05", category="agentic_development",
            trigger="We introduce Foo", capability="tool_use", autonomy_level="2",
            interaction="agent_tool", phase_hint="3", confidence="2")


def row(**kw):
    base = {c: "" for c in META + FIELDS}
    base.update(annotation_id="X-001", doc_id="d1", publication_date="2024-05-01",
                text_excerpt="We introduce Foo, an agent that browses.", annotator="HA")
    base.update(kw)
    return base


def check(**kw):
    return validate(pd.DataFrame([row(**kw)]), TAX)


def test_clean():
    assert _clean(None) == ""
    assert _clean("1.0") == "1"
    assert _clean(" NA ") == "NA"
    assert _clean("2023-05-01 00:00:00") == "2023-05-01"


def test_valid_row_has_no_errors():
    assert (check(**GOOD)["severity"] == "ERROR").sum() == 0


def test_missing_category_is_error():
    issues = check(**{**GOOD, "category": ""})
    assert ((issues["field"] == "category") & (issues["severity"] == "ERROR")).any()


def test_bad_date_is_error():
    issues = check(**{**GOOD, "event_date": "05/2024"})
    assert ((issues["field"] == "event_date") & (issues["severity"] == "ERROR")).any()


def test_trigger_not_in_text_is_warning():
    issues = check(**{**GOOD, "trigger": "nonsense phrase"})
    assert ((issues["field"] == "trigger") & (issues["severity"] == "WARN")).any()


def test_non_event_with_filled_fields_warns():
    issues = check(has_event="no_off_topic", category="agentic_development")
    assert (issues["severity"] == "WARN").any()
    assert (issues["severity"] == "ERROR").sum() == 0


def test_blank_has_event_is_error():
    assert (check()["severity"] == "ERROR").sum() == 1


def test_assign_holdout_is_deterministic_and_sized():
    ev = pd.DataFrame({"category": ["agentic_development"] * 10, "x": range(10)})
    a = assign_holdout(ev, 0.2, 42)
    b = assign_holdout(ev, 0.2, 42)
    assert (a["classifier_split"] == "holdout").sum() == 2
    assert a["classifier_split"].tolist() == b["classifier_split"].tolist()
