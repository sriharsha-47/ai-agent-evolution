"""Split the 150-document threshold sample between labellers, then merge the labelled parts back.

  python scripts/threshold_sample_tools.py split --names HA SH VV TS     # writes threshold_sample_<NAME>.xlsx
  python scripts/threshold_sample_tools.py merge --names HA SH VV TS     # writes threshold_sample_labelled.xlsx

Each labeller fills ONLY column H (has_event): yes / no_not_significant / no_off_topic / no_unclear_text,
using docs/annotation_guidelines.md section 3. Rows are dealt round-robin within each source type, so everyone
gets a mix of blogs, papers and release notes.
"""
import argparse
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "annotations"
SRC = ANN / "threshold_sample.xlsx"
ALLOWED = {"yes", "no_not_significant", "no_off_topic", "no_unclear_text"}


def split(names):
    ws = openpyxl.load_workbook(SRC)["Sample"]
    hdr = [c.value for c in ws[1]]
    st = hdr.index("source_type") + 1
    by = {}
    for r in range(2, ws.max_row + 1):
        by.setdefault(ws.cell(r, st).value, []).append(r)
    assign = {n: set() for n in names}
    k = 0
    for rows in by.values():
        for r in rows:
            assign[names[k % len(names)]].add(r)
            k += 1
    for n, keep in assign.items():
        wb = openpyxl.load_workbook(SRC)
        s = wb["Sample"]
        for r in sorted((r for r in range(2, s.max_row + 1) if r not in keep), reverse=True):
            s.delete_rows(r)
        wb.save(ANN / f"threshold_sample_{n}.xlsx")
        print(f"{n}: {len(keep)} documents -> threshold_sample_{n}.xlsx")


def merge(names):
    wb = openpyxl.load_workbook(SRC)
    ws = wb["Sample"]
    hdr = [c.value for c in ws[1]]
    ci, hi, ni = hdr.index("doc_id") + 1, hdr.index("has_event") + 1, hdr.index("notes") + 1
    pos = {ws.cell(r, ci).value: r for r in range(2, ws.max_row + 1)}
    problems, done = [], set()
    for n in names:
        s = openpyxl.load_workbook(ANN / f"threshold_sample_{n}.xlsx")["Sample"]
        h = [c.value for c in s[1]]
        for row in s.iter_rows(min_row=2, values_only=True):
            d = dict(zip(h, row))
            lab = str(d["has_event"] or "").strip()
            if lab not in ALLOWED:
                problems.append((n, d["doc_id"], lab or "blank"))
                continue
            r = pos[d["doc_id"]]
            ws.cell(r, hi).value = lab
            ws.cell(r, ni).value = d.get("notes") or None
            done.add(d["doc_id"])
    if problems:
        raise SystemExit(f"{len(problems)} rows have a missing/invalid has_event (first: {problems[:5]}).")
    if len(done) != len(pos):
        raise SystemExit(f"Only {len(done)} of {len(pos)} documents are labelled.")
    out = ANN / "threshold_sample_labelled.xlsx"
    wb.save(out)
    print(f"wrote {out} with {len(done)} labelled documents")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["split", "merge"])
    ap.add_argument("--names", nargs="+", default=["HA", "SH", "VV", "TS"])
    a = ap.parse_args()
    (split if a.cmd == "split" else merge)(a.names)


if __name__ == "__main__":
    main()
