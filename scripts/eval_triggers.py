"""Score trigger rules against the gold annotation sheet.
Usage: python scripts/eval_triggers.py <corpus.jsonl> <annotation_xlsx> [--show-failures]
Gold = rows with has_event == yes and a trigger string that occurs in the document text.
Hit levels: span = candidate phrase overlaps the gold trigger; sentence = candidate sentence contains the gold trigger.
"""
import collections, json, sys
from pathlib import Path
import openpyxl
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.extraction.triggers import find_triggers

TR = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‑": "-", "‐": "-", " ": " "})
corpus = {}
for line in open(sys.argv[1], encoding="utf-8"):
    d = json.loads(line); corpus[d["doc_id"]] = d
wb = openpyxl.load_workbook(sys.argv[2], read_only=True)
rows = []
for sheet in ("Pilot", "Batch_SH"):
    it = wb[sheet].iter_rows(values_only=True); hdr = next(it)
    rows += [dict(zip(hdr, r)) for r in it]

def overlap(a0, a1, b0, b1): return a0 < b1 and b0 < a1
res = collections.defaultdict(list); fails = []; scores = {"yes": [], "no_not_significant": [], "no_off_topic": []}
for r in rows:
    d = corpus[r["doc_id"]]; st = d["source_type"]; text = d["text"]
    c = find_triggers(text, st)
    scores[r["has_event"]].append(c[0]["score"] if c else 0.0)
    if r["has_event"] != "yes" or not r["trigger"]:
        continue
    i = text.translate(TR).find(r["trigger"].translate(TR))
    if i < 0:
        continue
    g0, g1 = i, i + len(r["trigger"])
    for k in (1, 3):
        top = c[:k]
        res[(st, k, "span")].append(any(overlap(x["start"], x["end"], g0, g1) for x in top))
        res[(st, k, "sentence")].append(any(x["sent_start"] <= g0 < x["sent_end"] for x in top))
        res[("all", k, "span")].append(res[(st, k, "span")][-1]); res[("all", k, "sentence")].append(res[(st, k, "sentence")][-1])
    if not res[(st, 3, "sentence")][-1]:
        fails.append((r["annotation_id"], st, r["trigger"][:60], [x["trigger"][:50] for x in c[:2]]))

print("GOLD TRIGGER HIT RATE (hit/total)")
for st in ("all", "lab_blog", "release_notes", "paper"):
    cells = []
    for lvl in ("span", "sentence"):
        for k in (1, 3):
            v = res[(st, k, lvl)]; cells.append(f"top{k}-{lvl} {sum(v)}/{len(v)}")
    print(f"  {st:14}", " | ".join(cells))
def auc(pos, neg):
    w = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg); return w / (len(pos) * len(neg))
neg = scores["no_not_significant"] + scores["no_off_topic"]
print(f"\nEVENT-SCORE AUC yes({len(scores['yes'])}) vs no({len(neg)}): {auc(scores['yes'], neg):.2f}"
      f"  | vs not_significant only: {auc(scores['yes'], scores['no_not_significant']):.2f}"
      f"  | vs off_topic only: {auc(scores['yes'], scores['no_off_topic']):.2f}")
print("mean score yes %.2f / not_sig %.2f / off_topic %.2f" % tuple(sum(scores[k]) / len(scores[k]) for k in ("yes", "no_not_significant", "no_off_topic")))
if "--show-failures" in sys.argv:
    print("\nMISSED (gold trigger sentence not in top-3):")
    for f in fails: print(" ", f)
