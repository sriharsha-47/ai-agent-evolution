"""D12: autonomy_level (0-4/NA) and interaction (6 types) predictors: keyword rules + helpers for a TF-IDF model.

Rules are written from the guideline definitions (scales in configs/taxonomy.yaml), not tuned on the gold.
"""
from __future__ import annotations
import re

_AUT = [
    ("4", r"continuous(?:ly)?|persistent|self-directed|open-ended|lifelong|long-running|sets? (?:its own )?(?:sub-?)?goals?|runs? 24/7"),
    ("3", r"autonomous(?:ly)?|end-to-end|without (?:any )?human|no human|fully automat\w+|completes? (?:the )?(?:whole )?tasks?"),
    ("2", r"multi-step|plans? and|planning|human[- ]in[- ]the[- ]loop|approv\w+|supervis\w+|iterative(?:ly)?|workflow|agents? (?:collaborat|coordinat)"),
    ("1", r"function[- ]call\w*|tool[- ]?use|\bplugins?\b|\bAPI call|calls? (?:an |the )?(?:tool|API)"),
]
_NA = re.compile(r"benchmark|survey|dataset|policy|bounty|partnership|funding|grants?|agreement|nonprofit|study of|how people", re.I)
_AUT_RX = [(lvl, re.compile(p, re.I)) for lvl, p in _AUT]

_INT = [
    ("agent_agent", r"multi-?agent|agents? (?:collaborat|coordinat|communicat|negotiat|talk)|orchestrat|delegat|group chat|swarm"),
    ("agent_environment", r"environment|simulat\w+|embodied|\bgames?\b|minecraft|robot\w*|sandbox|open-world|real[- ]world"),
    ("ecosystem", r"ecosystem|platform|marketplace|developers|partnership|enterprise|customers|rollout|integrations?|protocol|\bSDK\b|\bcluster\b"),
    ("agent_tool", r"\btools?\b|function[- ]call\w*|\bAPIs?\b|plugin|\bMCP\b|browse|code interpreter|retriev\w+"),
    ("human_agent", r"\bagents?\b|assistant|copilot|delegate|autonomous"),
    ("human_model", r"language model|chat\w*|prompt\w*|\bGPT\b|\bmodel\b|completion"),
]
_INT_RX = [(c, re.compile(p, re.I)) for c, p in _INT]


def _head(doc: dict) -> str:
    return (doc.get("title") or "") + ". " + (doc.get("text") or "")[:900]


def rule_autonomy(doc: dict) -> str:
    t = _head(doc)
    for lvl, rx in _AUT_RX:
        if rx.search(t):
            return lvl
    return "NA" if _NA.search(t) else "0"


def rule_interaction(doc: dict) -> str:
    t = _head(doc)
    best, best_n = "human_model", 0
    for c, rx in _INT_RX:
        n = len(rx.findall(t))
        if n > best_n:
            best, best_n = c, n
    return best
