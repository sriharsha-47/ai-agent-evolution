from ainevo.extraction.triggers import find_triggers, event_score


def top(text, st=None):
    c = find_triggers(text, st)
    return c[0] if c else None


def test_blog_lead_announcement_is_top():
    c = top("Today, we’re introducing OpenAI for Healthcare. Many hospitals use AI. Earlier we launched a pilot.", "lab_blog")
    assert "introducing" in c["trigger"] and c["rule"] == "announce"


def test_paper_we_propose():
    c = top("Agents are hard to evaluate. We propose Foo, a benchmark for agents.", "paper")
    assert c["rule"] == "research" and "propose" in c["trigger"]


def test_noun_use_is_not_a_trigger():
    assert top("The launch of ChatGPT in 2022 changed things.", "lab_blog") is None


def test_release_note_feature_beats_fix():
    c = top("## Features - Add CheckpointConfig for checkpointing ## Bug Fixes - Fixed flaky test", "release_notes")
    assert c["trigger"].startswith("Add CheckpointConfig")


def test_maintenance_only_note_has_zero_score():
    assert event_score("## Fixes - Fixed flaky test - Bump dependency versions - chore(core): cleanup", "release_notes") == 0.0


def test_offsets_point_into_original_text():
    t = "Read paper Loading… Share Today, we’re introducing Foo."
    c = top(t, "lab_blog")
    assert t[c["verb_start"]:c["verb_end"]].lower().startswith("introducing")
