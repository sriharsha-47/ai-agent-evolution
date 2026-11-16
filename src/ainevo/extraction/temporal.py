"""Temporal extraction: find dates / time phrases in a document and turn them into real dates.

Output format follows the annotation guideline: YYYY-MM-DD, or YYYY-MM if only the month is known.
Relative phrases ("last week", "in March") are resolved against the document's publication date.
"""
from __future__ import annotations
import re
from datetime import date, datetime


MONTHS = ("January|February|March|April|May|June|July|August|September|October|November|December|"
          "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sept|Sep|Oct|Nov|Dec")
_M = rf"(?:{MONTHS})\.?"

# Absolute patterns, tried in order of precision. Each yields (regex, granularity).
_PATTERNS = [
    ("iso_day",   re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"), "day"),
    ("mdy",       re.compile(rf"\b({_M})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b"), "day"),
    ("dmy",       re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_M}),?\s+(\d{{4}})\b"), "day"),
    ("iso_month", re.compile(r"\b(\d{4})-(\d{2})\b(?!-\d)"), "month"),
    ("my",        re.compile(rf"\b({_M})\s+(\d{{4}})\b"), "month"),
]
# Phrases that spaCy tags DATE but that are NOT usable event dates.
_RELATIVE = re.compile(
    r"\b(today|yesterday|tomorrow|last (?:week|month|year)|this (?:week|month|year)|"
    r"(?:\w+) (?:days?|weeks?|months?|years?) ago|earlier this \w+|next (?:week|month|year))\b", re.I)
_BARE_YEAR = re.compile(r"^(?:in\s+|since\s+|by\s+|during\s+)?((?:19|20)\d{2})$", re.I)
_BARE_MONTH = re.compile(rf"^(?:in\s+|since\s+|by\s+|during\s+|early\s+|late\s+|mid-?\s*)?({_M})$", re.I)


def _fmt(y: int, m: int | None = None, d: int | None = None) -> str | None:
    try:
        if d:
            return date(y, m, d).isoformat()
        if m:
            date(y, m, 1)
            return f"{y:04d}-{m:02d}"
    except ValueError:
        return None
    return f"{y:04d}"


def _month_num(name: str) -> int:
    return datetime.strptime(name.strip(".")[:3], "%b").month


def regex_dates(text: str) -> list[dict]:
    """High-precision absolute dates found by regex. Overlapping matches keep the most specific one."""
    found = []
    for kind, rx, gran in _PATTERNS:
        for m in rx.finditer(text):
            g = m.groups()
            if kind == "iso_day":
                norm = _fmt(int(g[0]), int(g[1]), int(g[2]))
            elif kind == "mdy":
                norm = _fmt(int(g[2]), _month_num(g[0]), int(g[1]))
            elif kind == "dmy":
                norm = _fmt(int(g[2]), _month_num(g[1]), int(g[0]))
            elif kind == "iso_month":
                norm = _fmt(int(g[0]), int(g[1]))
            else:
                norm = _fmt(int(g[1]), _month_num(g[0]))
            if norm:
                found.append({"text": m.group(0), "start": m.start(), "end": m.end(), "norm": norm,
                              "granularity": gran, "kind": "absolute", "source": f"regex:{kind}"})
    found.sort(key=lambda d: (d["start"], -(d["end"] - d["start"])))
    kept = []
    for d in found:  # drop matches contained in an earlier (longer) one
        if not any(d["start"] >= k["start"] and d["end"] <= k["end"] for k in kept):
            kept.append(d)
    return kept


def resolve_phrase(phrase: str, pub_date: str | None) -> dict | None:
    """Resolve a spaCy DATE phrase against the publication date. Returns {norm, granularity, kind} or None."""
    p = phrase.strip()
    base = datetime.fromisoformat(pub_date) if pub_date else None
    m = _BARE_YEAR.match(p)
    if m:
        return {"norm": m.group(1), "granularity": "year", "kind": "absolute"}
    m = _BARE_MONTH.match(p)
    if m and base:  # "in March" -> most recent March on or before the publication date
        mon = _month_num(m.group(1))
        year = base.year if mon <= base.month else base.year - 1
        return {"norm": f"{year:04d}-{mon:02d}", "granularity": "month", "kind": "relative"}
    if _RELATIVE.search(p) and base:
        import dateparser  # lazy: regex_dates() must work even where dateparser's compiled deps are blocked
        dt = dateparser.parse(p, settings={"RELATIVE_BASE": base, "PREFER_DATES_FROM": "past", "TIMEZONE": "UTC", "RETURN_AS_TIMEZONE_AWARE": False})
        if dt:
            return {"norm": dt.date().isoformat(), "granularity": "day", "kind": "relative"}
    return None


def extract_dates(text: str, pub_date: str | None = None, nlp=None) -> list[dict]:
    """All dates in `text`: regex absolute dates + resolvable spaCy DATE phrases. Sorted by position."""
    out = regex_dates(text)
    if nlp is not None:
        for ent in nlp(text).ents:
            if ent.label_ != "DATE":
                continue
            if any(ent.start_char < o["end"] and ent.end_char > o["start"] for o in out):
                continue
            r = resolve_phrase(ent.text, pub_date)
            if r:
                out.append({"text": ent.text, "start": ent.start_char, "end": ent.end_char,
                            "source": "spacy:DATE", **r})
    out.sort(key=lambda d: d["start"])
    return out


def best_event_date(dates: list[dict], pub_date: str | None) -> tuple[str | None, str]:
    """Pick one event date following the guideline: the first specific absolute date not after publication,
    otherwise fall back to the publication date. Returns (date, how)."""
    pub = pub_date or ""
    for d in dates:
        if d["kind"] == "absolute" and d["granularity"] in ("day", "month") and (not pub or d["norm"][:7] <= pub[:7]):
            return d["norm"], "text"
    return (pub_date, "publication_date_fallback") if pub_date else (None, "none")
