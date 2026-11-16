from ainevo.extraction.segment import split_sentences


def texts(t):
    return [s["text"] for s in split_sentences(t)]


def test_splits_sentences_and_keeps_abbreviations():
    out = texts("We trained GPT-2, e.g. a small model. It works! See Fig. 2 for details.")
    assert out == ["We trained GPT-2, e.g. a small model.", "It works!", "See Fig. 2 for details."]


def test_release_note_bullets_and_headings_split():
    out = texts("## Features - Add CheckpointConfig for checkpointing - Fixed flaky test")
    assert out == ["Features", "Add CheckpointConfig for checkpointing", "Fixed flaky test"]


def test_offsets_point_into_original_text():
    t = "Alpha beta. Gamma delta."
    for s in split_sentences(t):
        assert t[s["start"]:s["end"]] == s["text"]


# ---- event segmentation -------------------------------------------------------------------------------------------
from ainevo.extraction.segment import segment_events


def test_release_notes_features_become_units_fixes_do_not():
    t = "## What's New ### Features - Add CheckpointConfig for checkpointing - Support nested `Team` as a participant ## Bug Fixes - Fixed flaky test"
    u = segment_events(t, "release_notes")
    assert [x["text"].split()[0] for x in u] == ["Add", "Support"]
    assert all(x["n_sentences"] == 1 for x in u)          # release-note bullets get no follow-on context


def test_blog_unit_includes_context_and_stops_at_next_event():
    t = "Today, we’re introducing Atlas, a browser agent. It runs web tasks. Paid users get it first. Separately, we launched Beta on June 1, 2025."
    u = segment_events(t, "lab_blog")
    assert len(u) == 2 and u[0]["n_sentences"] == 3 and u[1]["text"].startswith("Separately")
    assert u[1]["dates"] == ["2025-06-01"]


def test_exactly_one_primary_unit_and_valid_offsets():
    t = "Today, we’re introducing Atlas, a browser agent. Later, we released Beta to everyone."
    u = segment_events(t, "lab_blog")
    assert sum(x["is_primary"] for x in u) == 1
    for x in u:
        assert " ".join(t[x["start"]:x["end"]].split()) == x["text"]


def test_no_trigger_gives_no_units():
    assert segment_events("The weather is nice today. Agents are interesting.", "lab_blog") == []
