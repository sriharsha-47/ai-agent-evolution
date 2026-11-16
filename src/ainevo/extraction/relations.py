"""D9: event-level relation extraction: (organisation, system, capability, action).

organisation / system come from ner.guess_organizations / guess_systems, the action phrase is extracted from the primary
trigger unit, the capability tag comes from a keyword-rule baseline (`rule_capability`) or any fitted model passed in
(`capability_model.predict([text])`). No spaCy needed.
"""
from __future__ import annotations
import re

from .ner import guess_organizations, guess_systems
from .pipeline import extract_document

_CAP_RULES: list[tuple[str, str]] = [
    ("computer_use", r"computer use|\bGUI\b|desktop|screenshot|operat\w+ a computer|web browser"),
    ("multi_agent_coordination", r"multi-?agent|agents? (?:collaborat|coordinat|communicat|negotiat)|orchestrat|delegat|group chat|\bcrew\b|swarm"),
    ("safety_control", r"safety|alignment|jailbreak|moderation|red[- ]team|misuse|harm|bounty|backdoor|adversarial|robustness|preferences|\bRLHF\b|security"),
    ("evaluation_benchmark", r"benchmark|evaluat\w+|leaderboard|dataset|test suite"),
    ("memory", r"\bmemory\b|remember|retriev|long-term context|knowledge base|\bRAG\b"),
    ("coding", r"\bcode\b|coding|software|programming|\bSWE\b|pull request|compiler|Codex"),
    ("multimodal", r"image|vision|audio|video|speech|multimodal|\bvoice\b|Sora|DALL"),
    ("tool_use", r"\btools?\b|function[- ]call|\bAPIs?\b|plugin|\bMCP\b|integrat\w+|connector"),
    ("planning", r"\bplan(?:s|ning|ner)?\b|long-horizon|multi-step|workflow|task decomposition"),
    ("reasoning", r"reasoning|chain[- ]of[- ]thought|\bmath\b|\bproof\b|\bo1\b"),
    ("infrastructure", r"platform|\bSDK\b|protocol|infrastructure|cluster|compute|runtime|deployment|enterprise|partnership|release notes|version"),
    ("text_generation", r"language model|text generation|\bGPT-?[234]\b|summariz"),
]
_CAP_RX = [(c, re.compile(p, re.I)) for c, p in _CAP_RULES]


def rule_capability(doc: dict) -> str:
    """Keyword baseline on title + opening. Highest number of hits wins; ties go to the earlier (more specific) rule."""
    t = (doc.get("title") or "") + ". " + (doc.get("text") or "")[:700]
    best, best_n = "other", 0
    for cap, rx in _CAP_RX:
        n = len(rx.findall(t))
        if n > best_n:
            best, best_n = cap, n
    return best


def action_phrase(unit_text: str, trigger: str | None, max_words: int = 22) -> str:
    """Short extractive action: the clause after the trigger in the trigger sentence, else the sentence itself."""
    s = re.sub(r"\s+", " ", unit_text or "").strip()
    first = re.split(r"(?<=[.!?])\s+(?=[A-Z])", s, maxsplit=1)[0]
    if trigger:
        i = first.lower().find(trigger.lower())
        if i >= 0:
            tail = first[i:]
            if len(tail.split()) >= 5:
                first = tail
    words = first.split()
    return " ".join(words[:max_words]).rstrip(",;:")


def extract_relation(doc: dict, capability_model=None) -> dict:
    rec = extract_document(doc)
    prim = next((u for u in rec["units"] if u["is_primary"]), rec["units"][0] if rec["units"] else None)
    cap = capability_model.predict([doc.get("title", "") + ". " + (doc.get("text") or "")[:3000]])[0] if capability_model else rule_capability(doc)
    return {"doc_id": doc["doc_id"], "organization": rec["organizations"], "system": rec["systems"],
            "capability": cap, "action": action_phrase(prim["text"], prim["trigger"]) if prim else "",
            "unit_id": prim["unit_id"] if prim else None}
