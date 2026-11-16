"""Sentence segmentation without spaCy (the lab laptop blocks spaCy's compiled files).

The cleaned corpus has no newlines, so release-note bullets (" - Fixed ...") and headings (" ## Features") appear inline
and must be treated as boundaries as well as ". ". Returns spans with character offsets into the original text.
"""
from __future__ import annotations
import re

_ABBREV = {"e.g", "i.e", "et al", "etc", "vs", "fig", "figs", "eq", "no", "dr", "mr", "mrs", "ms", "prof", "approx",
           "inc", "ltd", "u.s", "u.k", "cf", "al", "ca", "sec", "st"}
_END = re.compile(r"[.!?]+[\"'’”)\]]*\s+(?=[\"'“‘(\[`]?[A-Z0-9`])")
_BULLET = re.compile(r"\s(?:[-*•]|#{2,4})\s+(?=[A-Z`\[(\"“]|[a-z]+(?:ed|s)?\s)")
_HEADING = re.compile(r"\s#{2,4}\s")


def _is_abbrev(text: str, end_idx: int) -> bool:
    m = re.search(r"([A-Za-z][A-Za-z.]{0,6})[.]$", text[max(0, end_idx - 9):end_idx])
    return bool(m) and m.group(1).lower().rstrip(".") in _ABBREV


def split_sentences(text: str) -> list[dict]:
    """[{start, end, text}] covering the non-blank text. Boundaries: sentence punctuation, bullets, headings."""
    cuts = {0, len(text)}
    for m in _END.finditer(text):
        if not _is_abbrev(text, m.start() + 1):
            cuts.add(m.end())
    for m in _BULLET.finditer(text):
        cuts.add(m.start() + 1)
    for m in _HEADING.finditer(text):
        cuts.add(m.start() + 1)
    pts = sorted(cuts)
    out = []
    for a, b in zip(pts, pts[1:]):
        seg = text[a:b]
        lead = len(seg) - len(seg.lstrip(" -*•#"))
        seg2 = seg.strip(" -*•#\t")
        if seg2:
            out.append({"start": a + lead, "end": a + lead + len(seg2), "text": seg2})
    return out


# --------------------------------------------------------------------------------------------------------------------
# Event segmentation: cut a document into candidate event units (trigger sentence + the sentences that elaborate it).
# --------------------------------------------------------------------------------------------------------------------
_SHORT_HEADING = re.compile(r"^[^.!?]{0,44}$")


def segment_events(text: str, source_type: str | None = None, min_score: float = 3.0, max_units: int = 5,
                   follow: int = 2) -> list[dict]:
    """Candidate event units in document order.

    Unit = a sentence holding a trigger scoring >= min_score, plus up to `follow` following sentences (blogs, papers),
    stopping early at a heading or at the next trigger sentence (a new event). Release notes: the bullet alone.
    Each unit: {unit_id, start, end, text, trigger, trigger_start, trigger_end, rule, score, section, n_sentences,
    dates, rank, is_primary}. `rank` 1 = highest score (ties: earliest); only the top `max_units` units are kept.
    """
    from .triggers import find_triggers, _mask_chrome   # lazy: triggers imports this module
    from .temporal import regex_dates

    sents = split_sentences(_mask_chrome(text))
    if not sents:
        return []
    best = {}                                            # sentence index -> best candidate in that sentence
    for c in find_triggers(text, source_type):
        if c["score"] >= min_score and (c["sent_idx"] not in best or c["score"] > best[c["sent_idx"]]["score"]):
            best[c["sent_idx"]] = c
    if not best:
        return []
    n_follow = 0 if source_type == "release_notes" else follow
    units = []
    for i in sorted(best):
        j = i
        while j + 1 < len(sents) and j - i < n_follow:
            nxt = sents[j + 1]
            if (j + 1) in best or _SHORT_HEADING.match(nxt["text"]):
                break
            j += 1
        c = best[i]
        a, b = sents[i]["start"], sents[j]["end"]
        unit_text = re.sub(r"\s+", " ", text[a:b]).strip()
        units.append({"start": a, "end": b, "text": unit_text, "trigger": c["trigger"], "trigger_start": c["start"],
                      "trigger_end": c["end"], "rule": c["rule"], "score": c["score"], "section": c["section"],
                      "n_sentences": j - i + 1, "dates": [d["norm"] for d in regex_dates(text[a:b])]})
    order = sorted(range(len(units)), key=lambda k: (-units[k]["score"], units[k]["start"]))
    keep = set(order[:max_units])
    out = []
    for k, u in enumerate(units):
        if k in keep:
            u["rank"] = order.index(k) + 1
            u["is_primary"] = u["rank"] == 1
            out.append(u)
    for n, u in enumerate(out, 1):
        u["unit_id"] = n
    return out
