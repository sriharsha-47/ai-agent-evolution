"""Named-entity extraction: spaCy NER plus a small rule list for AI labs and model/agent names.

The stock spaCy model does not know most AI model and framework names (GPT-4o, CrewAI, ...), so a regex
gazetteer runs on top and wins on overlap. Labels used downstream: ORG, MODEL_AGENT, PERSON.
"""
from __future__ import annotations
import re

ORGS = ["OpenAI", "Anthropic", "Google DeepMind", "DeepMind", "Google Research", "Google", "Meta AI", "Meta",
        "Microsoft Research", "Microsoft", "Mistral AI", "Mistral", "DeepSeek", "Alibaba", "Qwen", "xAI", "Cohere",
        "AI21 Labs", "Hugging Face", "Stability AI", "NVIDIA", "Apple", "Amazon", "AWS", "Stanford", "MIT",
        "Carnegie Mellon University", "UC Berkeley", "Tsinghua University", "Peking University", "Allen Institute for AI",
        "crewAIInc", "LangChain", "Significant-Gravitas"]
# model / agent / framework names: (regex, canonical label kept as matched text)
MODEL_PATTERNS = [
    r"\bGPT[-‑‐]?\d(?:\.\d+)?(?:[- ]?(?:o|mini|nano|turbo|codex|oss(?:-\d+b)?|Turbo))*\b",
    r"\bo[1-9](?:-(?:mini|preview|pro))?\b(?=\s+(?i:model|series|system))",
    r"(?<=OpenAI )o[1-9](?:-(?:mini|preview|pro))?\b", r"\bo[1-9]-(?:mini|preview|pro)\b",
    r"\bChatGPT\b", r"\bCodex\b", r"\bSora(?: \d)?\b", r"\bDALL[·-]?E(?: \d)?\b", r"\bWhisper\b", r"\bCLIP\b",
    r"\bClaude(?: (?:Opus|Sonnet|Haiku))?(?: \d(?:\.\d+)?)?\b", r"\bGemini(?: \d(?:\.\d+)?)?(?: (?:Pro|Flash|Ultra))?\b",
    r"\bLlama[- ]?\d(?:\.\d+)?(?:-\d+B)?\b", r"\bLLaMA\b", r"\bMistral[- ]?\d*B?\b", r"\bQwen\d?(?:\.\d+)?\b", r"\bDeepSeek-?(?:V\d|R\d)\b",
    r"\bKimi(?: K\d)?\b", r"\bGrok(?:-?\d)?\b", r"\bBERT\b", r"\bRoBERTa\b", r"\bT5\b", r"\bAuto-?GPT\b", r"\bAutoGen(?: Studio)?\b", r"\bCrewAI\b", r"\bLangChain\b",
    r"\bLangGraph\b", r"\bLlamaIndex\b", r"\bMCP\b", r"\bModel Context Protocol\b", r"\bA2A\b", r"\bReAct\b", r"\bSWE-bench\b",
]
_GAZ = [(re.compile(p), "MODEL_AGENT") for p in MODEL_PATTERNS] + \
       [(re.compile(r"(?<![\w-])" + re.escape(o) + r"(?![\w-])"), "ORG") for o in sorted(ORGS, key=len, reverse=True)]
KEEP_SPACY = {"ORG", "PERSON", "PRODUCT", "GPE"}
_JUNK = re.compile(r"^(?:the |a |an )?(?:LLMs?|RL|AI|ML|NLP|API|APIs|GUI|MDPs?|POMDPs?|SOTA|NPU|NPUs|VLMs?|LLM-based|CoT)$", re.I)


def load_nlp(model: str = "en_core_web_sm"):
    import spacy  # lazy: spaCy is blocked on some Windows laptops, the rule helpers below must still import
    return spacy.load(model, disable=["lemmatizer"])


def extract_entities(text: str, nlp, title: str | None = None) -> list[dict]:
    """Entities as dicts {text,label,start,end,source}; rule matches override overlapping spaCy entities.

    Rules run on "title. text" (release-note titles carry the org and version); the statistical spaCy model runs on
    the body text only, because paper titles are title-case and get mis-tagged as ORG. Offsets refer to the string
    "title. text" when a title is given, otherwise to `text`.
    """
    prefix = (title + ". ") if title else ""
    full = prefix + text
    rules = []
    for rx, label in _GAZ:
        for m in rx.finditer(full):
            rules.append({"text": m.group(0), "label": label, "start": m.start(), "end": m.end(), "source": "rules"})
    rules.sort(key=lambda e: (e["start"], -(e["end"] - e["start"])))
    kept_rules = []
    for e in rules:
        if not any(e["start"] < k["end"] and e["end"] > k["start"] for k in kept_rules):
            kept_rules.append(e)
    out = list(kept_rules)
    for ent in nlp(text).ents:
        if ent.label_ not in KEEP_SPACY or _JUNK.match(ent.text.strip()):
            continue
        s0, e0 = ent.start_char + len(prefix), ent.end_char + len(prefix)
        if any(s0 < k["end"] and e0 > k["start"] for k in kept_rules):
            continue
        out.append({"text": ent.text, "label": ent.label_, "start": s0, "end": e0, "source": "spacy"})
    return sorted(out, key=lambda e: e["start"])


# ---------------------------------------------------------------------------------------------------------------
# Task 5 additions: document-level organisation / system-name guesses (no spaCy needed)
# ---------------------------------------------------------------------------------------------------------------
_SOURCE_ORG = {"openai": "OpenAI"}
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2212"), "-")


def canon(s: str) -> str:
    """ASCII hyphens and spaces: the corpus mixes U+2011 / U+2010 and non-breaking spaces ("GPT\u20114" vs "GPT-4")."""
    return s.translate(_DASHES).replace("\u00a0", " ").replace("\u202f", " ")


def owner_from_source(source: str | None, source_type: str | None) -> str | None:
    """Organisation implied by the corpus `source` field.

    Lab blogs: the lab itself ("OpenAI"). Release notes: the GitHub owner ("microsoft/autogen" -> "microsoft").
    Papers: None, because the affiliation is not in the abstract and an author list is not an organisation event actor.
    """
    if not source or source_type == "paper":
        return None
    key = source.strip().lower()
    if key in _SOURCE_ORG:
        return _SOURCE_ORG[key]
    if "/" in source:
        return source.split("/")[0]
    return source.strip()


_VERSION = re.compile(r"\b[vV]?\d+(?:\.\d+)+(?:[a-z]+\d*)?\b|\bpython-v[\d.]+\b")
_LEAD = re.compile(r"^(?:Introducing|Launching|Announcing|Meet|Upgrading|Scaling|Improving|New)\s+", re.I)


def title_system_name(title: str | None, source_type: str | None) -> str | None:
    """Best single system / product name from the title, or None.

    paper: the prefix before ':' when it is short (<= 3 tokens), e.g. "ATLAS: Adaptive Trading ..." -> "ATLAS".
    release_notes: "<owner>/<repo> <tag>" -> "<repo> <tag>".
    lab_blog: first gazetteer model match inside the title.
    """
    if not title:
        return None
    t = canon(title.strip())
    if source_type == "paper":
        m = re.match(r"^([^:]{2,40}?)\s*:", t)
        if m and len(m.group(1).split()) <= 3:
            return m.group(1).strip()
        return None
    if source_type == "release_notes":
        m = re.match(r"^[\w.-]+/([\w.-]+)\s+(.*)$", t)
        if m:
            tag = m.group(2)
            return (m.group(1) + " " + tag).strip()
        return t
    hits = [m for rx, label in _GAZ if label == "MODEL_AGENT" for m in [rx.search(t)] if m]
    if hits:
        return min(hits, key=lambda m: (m.start(), -(m.end() - m.start()))).group(0)  # earliest, then longest
    return None


def gazetteer_entities(text: str, label: str) -> list[str]:
    """Distinct gazetteer matches of one label ("ORG" / "MODEL_AGENT") in order of first appearance, no spaCy."""
    seen, out = set(), []
    hits = []
    text = canon(text)
    for rx, lab in _GAZ:
        if lab != label:
            continue
        for m in rx.finditer(text):
            hits.append((m.start(), m.end(), m.group(0)))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    last_end = -1
    for st, en, g in hits:
        if st < last_end:
            continue
        last_end = en
        if g.lower() not in seen:
            seen.add(g.lower())
            out.append(g)
    return out


def guess_organizations(doc: dict) -> list[str]:
    """Source-derived organisation plus gazetteer organisations named in the title (partners, e.g. 'Amazon')."""
    out = []
    o = owner_from_source(doc.get("source"), doc.get("source_type"))
    if o:
        out.append(o)
    for g in gazetteer_entities(doc.get("title") or "", "ORG"):
        if g.lower() not in {x.lower() for x in out}:
            out.append(g)
    return out


def guess_systems(doc: dict, k_extra: int = 1) -> list[str]:
    """Title-derived system name first, then up to k_extra most frequent gazetteer model names from title + opening."""
    st = doc.get("source_type")
    out = []
    n = title_system_name(doc.get("title"), st)
    if n:
        out.append(n)
    if st != "release_notes":
        head = canon((doc.get("title") or "") + ". " + (doc.get("text") or "")[:600])
        counts: dict[str, int] = {}
        for g in gazetteer_entities(head, "MODEL_AGENT"):
            counts[g] = len(re.findall(re.escape(g), head))
        for g, _ in sorted(counts.items(), key=lambda kv: -kv[1])[:k_extra]:
            if g.lower() not in {x.lower() for x in out}:
                out.append(g)
    return out
