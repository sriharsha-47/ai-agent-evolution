"""Event-trigger rules: find the word/phrase that says something happened ("released", "we introduce", "Add ...").

Rule-based (no model needed). Each candidate gets a score; the document's event_score is its best candidate score.
Weights come from the annotation guideline's event types and are NOT yet tuned on held-out data.
"""
from __future__ import annotations
import re
from .segment import split_sentences

# (rule name, regex for the trigger verb/phrase, base weight)
RULES = [
    ("describe",  r"\b(?:our|this|the proposed)\s+(?:approach|method|framework|system|benchmark(?: suite)?|architecture|platform|study|work)\b[^.]{0,40}?\b(?:provides?|enables?|addresses|achieves?|improves?|introduces?|presents?|proposes?|offers?|supports?|evaluates?)\b", 2.0),
    ("announce",  r"\b(?:introduc(?:e|es|ed|ing)|releas(?:e|es|ed|ing)|launch(?:es|ed|ing)?|announc(?:e|es|ed|ing)|unveil(?:s|ed|ing)?|debut(?:s|ed)?|"
                  r"open-?sourc(?:e|es|ed|ing)|roll(?:s|ed|ing)? out|now (?:generally )?available|general availability|made available)\b", 3.0),
    ("research",  r"\b(?:we|this (?:paper|work|study)|our (?:paper|work))\s+(?:\w+\s+)?(?:present|propos|introduc|develop|design|demonstrat|show|build|built|train|trained|release|evaluate|investigat|study|report|find)\w*", 2.5),
    ("model",     r"\b(?:we(?:’|')ve|we have)\s+(?:\w+\s+)?(?:trained|developed|built|created|scaled|deployed|released)\b", 3.0),
    ("deal",      r"\b(?:partnership|partner(?:s|ed|ing)? with|agreement|co-?creat\w+|expands? (?:access|its)|will (?:invest|provide)|invest(?:s|ed|ment)? (?:of|in)|funding)\b", 2.0),
    ("scale",     r"\b(?:scaled|scales|scaling) (?:\w+\s+){0,3}to\b|\b(?:more than|over|nearly)\s+[\d.,]+\s*(?:million|billion|trillion|%)\s+(?:\w+\s+){0,2}(?:users|people|weekly|customers|classrooms|tokens)\b", 2.0),
    ("feature",   r"^(?:add(?:s|ed)?|introduc\w+|implement\w*|support\w*|enable\w*|expos\w+|allow\w*|launch\w*|new)\b", 2.0),
]
_RULES = [(n, re.compile(rx, re.I), w) for n, rx, w in RULES]
# Page chrome that is not content; replaced by spaces so character offsets are unchanged.
_CHROME = re.compile(r"Loading[….]*|\bShare\b(?=\s)|Read (?:newest version|paper|more|the full report here)\b|View model card|\(opens in a new window\)|[\u2060\u200b]", re.I)
# A match preceded by one of these is a noun use ("the launch of", "new release", "open-source datasets").
_NOUN_PREV = {"the", "a", "an", "of", "new", "full", "latest", "this", "next", "first", "each", "every", "its", "their", "our", "and", "or",
              "for", "with", "open", "since", "after", "before", "by"}
_NOUN_NEXT = re.compile(r"^[\s-]*(?:datasets?|models?|code|tools?|software|weights|version|notes?|candidate|test|fixes|mode|framework|libraries|library)\b", re.I)


_SEC_GOOD = re.compile(r"^(?:new )?(?:features?|what['’]s new|highlights?|added|enhancements?|new)\b", re.I)
_SEC_BAD = re.compile(r"^(?:bug ?fix\w*|fix(?:es|ed)?|chores?|docs?|documentation|dependenc\w+|maintenance|internal|tests?|ci|refactor\w*|core improvements)\b", re.I)


def _mask_chrome(text: str) -> str:
    return _CHROME.sub(lambda m: " " * len(m.group(0)), text)


def _noun_use(stext: str, m: re.Match, rule: str) -> bool:
    if rule != "announce":
        return False
    w = m.group(0).lower()
    base_form = re.fullmatch(r"(?:releas|launch|introduc|announc|unveil|debut|open-?sourc)e?", w) or w.startswith("open")
    prev = re.findall(r"[\w'’-]+", stext[:m.start()])[-1:] 
    if base_form and prev and prev[0].lower() in _NOUN_PREV:
        return True
    if w.startswith("open") and _NOUN_NEXT.match(stext[m.end():]):   # "open-source datasets/models" = adjective
        return True
    if base_form and _NOUN_NEXT.match(stext[m.end():]) and w.startswith(("releas", "launch")):
        return True
    return False
# Weak/negative cues
_MAINT = re.compile(r"^(?:fix(?:ed|es)?|bump(?:ed)?|chore|docs?|refactor\w*|update[sd]? (?:dep|doc|readme|version)|revert\w*|pin(?:ned)?|mock(?:ed)?|ignore[sd]?|test\w*|ci\b|lint\w*|typo\w*)", re.I)
_PRIOR = re.compile(r"\b(?:existing|previous|prior|recent(?:ly)?|traditional|current) (?:work|methods?|approaches|studies|systems?)\b|\bhas been\b|\bhave been\b|\bhowever\b|\bremains?\b", re.I)
_SUBJ = re.compile(r"\b(?:we(?:’|')?(?:re|ve)?|today|our)\b", re.I)


def _phrase(text: str, s: int, e: int, sent_end: int) -> tuple[int, int]:
    """Trigger phrase = adjacent subject ("Today, we’re" / "we") + the match + up to 5 following words, cut at punctuation."""
    pre = re.search(r"(?:today,?\s+)?(?:we(?:’|')?(?:re|ve)?\s+)$", text[max(0, s - 22):s], re.I)
    start = s - len(pre.group(0)) if pre else s
    tail = text[e:sent_end]
    m = re.match(r"(?:\s+[^\s,;:.()]+){0,5}", tail)
    return start, e + (m.end() if m else 0)


def find_triggers(text: str, source_type: str | None = None) -> list[dict]:
    """Ranked trigger candidates: [{trigger, start, end, verb_start, verb_end, sent_start, sent_end, rule, score}]."""
    text_scan = _mask_chrome(text)
    sents = split_sentences(text_scan)
    n = max(len(sents), 1)
    cands = []
    section = None                     # last release-note heading seen: "good" (features) / "bad" (fixes, chores) / None
    for i, sd in enumerate(sents):
        st, en, stext = sd["start"], sd["end"], sd["text"]
        if source_type == "release_notes":
            if len(stext) < 45 and _SEC_GOOD.match(stext):
                section = "good"
            elif len(stext) < 45 and _SEC_BAD.match(stext):
                section = "bad"
            if _MAINT.match(stext):
                continue
        for rule, rx, w in _RULES:
            for m in rx.finditer(stext):
                if rule == "feature" and m.start() != 0:
                    continue
                if _noun_use(stext, m, rule):
                    continue
                score = w
                score += 0.8 if _SUBJ.search(stext[:m.start() + 20]) else 0
                if source_type == "lab_blog":
                    score += 2.5 * 0.7 ** i                      # a blog states its news in the first sentences
                elif source_type == "release_notes":
                    score += {"good": 1.0, "bad": -1.0, None: 0.0}[section]
                else:
                    score += 1.0 * (1 - i / n)                   # papers: earlier sentences matter somewhat more
                score += 0.7 if len(stext) < 260 else 0
                if _PRIOR.search(stext):
                    score -= 1.5
                if source_type == "paper" and rule == "announce":
                    score -= 0.5                      # abstracts rarely "launch" things; keep research rule on top
                vs, ve = st + m.start(), st + m.end()
                ps, pe = _phrase(text_scan, vs, ve, en)
                cands.append({"trigger": re.sub(r"\s+", " ", text[ps:pe]).strip(), "start": ps, "end": pe, "verb_start": vs, "verb_end": ve,
                              "sent_start": st, "sent_end": en, "sent_idx": i, "section": section, "rule": rule, "score": round(score, 2)})
    cands.sort(key=lambda c: -c["score"])
    return cands


def event_score(text: str, source_type: str | None = None) -> float:
    c = find_triggers(text, source_type)
    return c[0]["score"] if c else 0.0
