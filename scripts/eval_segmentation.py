"""Evaluate event segmentation against the gold annotation sheet (one gold event per document).
Usage: python scripts/eval_segmentation.py <corpus.jsonl> <annotation_xlsx>
Metrics (events = has_event == yes):
  contain@1  primary unit contains the gold trigger     contain@any  any kept unit contains it
  coverage   gold `impact` numbers (>=2 digits or decimals) found in the primary unit vs. anywhere in the document
  units/doc  mean units kept for event docs vs. non-event docs (non-event docs should ideally have none)
"""
import collections, json, re, statistics, sys
from pathlib import Path
import openpyxl
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ainevo.extraction.segment import segment_events

TR = str.maketrans({"’": "'", "‘": "'", "“": '"', "”": '"', "‑": "-", "‐": "-", " ": " "})
corpus = {json.loads(l)["doc_id"]: json.loads(l) for l in open(sys.argv[1], encoding="utf-8")}
wb = openpyxl.load_workbook(sys.argv[2], read_only=True)
rows = []
for sheet in ("Pilot", "Batch_SH"):
    it = wb[sheet].iter_rows(values_only=True); hdr = next(it)
    rows += [dict(zip(hdr, r)) for r in it]
NUM = re.compile(r"\d+\.\d+|\d{2,}(?:,\d{3})*")
def nums(s): return {n.replace(",", "") for n in NUM.findall(s or "")}

def run(min_score, max_units, by_source=False):
    ev = collections.defaultdict(list); n_units = {"event": [], "non-event": []}
    c1 = ca = tot = docs_with = nonev_with = nonev = 0; cov_u = cov_d = cov_n = 0
    for r in rows:
        d = corpus[r["doc_id"]]; text = d["text"]
        units = segment_events(text, d["source_type"], min_score=min_score, max_units=max_units)
        if r["has_event"] != "yes":
            nonev += 1; nonev_with += bool(units); n_units["non-event"].append(len(units)); continue
        n_units["event"].append(len(units)); docs_with += bool(units)
        i = text.translate(TR).find((r["trigger"] or "").translate(TR))
        if i < 0: continue
        g0, g1 = i, i + len(r["trigger"]); tot += 1
        inside = lambda u: u["start"] <= g0 and g1 <= u["end"] + 1
        prim = [u for u in units if u["is_primary"]]
        c1 += any(inside(u) for u in prim); ca += any(inside(u) for u in units)
        key = d["source_type"]; ev[key].append((any(inside(u) for u in prim), any(inside(u) for u in units), len(units)))
        g = nums(r["impact"])
        if g:
            cov_n += 1; cov_d += g <= nums(text)
            cov_u += bool(prim) and g <= nums(prim[0]["text"])
    return dict(tot=tot, c1=c1, ca=ca, docs_with=docs_with, n_event=len(n_units["event"]), nonev=nonev, nonev_with=nonev_with,
                mean_ev=statistics.mean(n_units["event"]), mean_non=statistics.mean(n_units["non-event"]),
                cov_n=cov_n, cov_u=cov_u, cov_d=cov_d, ev=ev)

print("THRESHOLD CURVE (max_units=5)")
print(f"{'min_score':>9} | {'contain@1':>9} | {'contain@any':>11} | {'event docs w/ unit':>18} | {'non-event w/ unit':>17} | units/doc ev / non-ev")
for ms in (0, 3.0, 3.5, 4.0, 5.0):
    r = run(ms, 5)
    print(f"{ms:>9} | {r['c1']:>4}/{r['tot']:<4} | {r['ca']:>6}/{r['tot']:<4} | {r['docs_with']:>9}/{r['n_event']:<8} | {r['nonev_with']:>8}/{r['nonev']:<8} | {r['mean_ev']:.1f} / {r['mean_non']:.1f}")
r = run(3.0, 5)
print("\nBY SOURCE TYPE at min_score=3.0 (primary-contains / any-contains / mean units, n gold events)")
for k, v in sorted(r["ev"].items()):
    print(f"  {k:14} {sum(x[0] for x in v):>2}/{len(v):<2} | {sum(x[1] for x in v):>2}/{len(v):<2} | {statistics.mean(x[2] for x in v):.1f} units | n={len(v)}")
print(f"\nEVIDENCE COVERAGE (docs whose gold impact has numbers: {r['cov_n']}): numbers inside primary unit {r['cov_u']}/{r['cov_n']}; anywhere in doc {r['cov_d']}/{r['cov_n']}")
