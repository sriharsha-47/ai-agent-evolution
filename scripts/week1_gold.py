"""Week-1 workflow for HA. Run from the repo root:

  python scripts/week1_gold.py progress                     # how many rows each member has labelled
  python scripts/week1_gold.py validate --who HA            # check your own batch while you label
  python scripts/week1_gold.py validate --pilot             # check all four pilot sheets
  python scripts/week1_gold.py agreement                    # pilot kappa + disagreement list for the meeting
  python scripts/week1_gold.py qa-make                      # build blind QA sheets (10 docs each)
  python scripts/week1_gold.py qa-report                    # compare returned QA sheets with the key
  python scripts/week1_gold.py target                       # check split.yaml (+ label stats if gold set exists)
  python scripts/week1_gold.py merge                        # build gold_events_v1.csv (Day 7)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from ainevo.evaluation import gold  # noqa: E402
from ainevo.forecasting import splits  # noqa: E402

ROOT = gold.ROOT
ANN = ROOT / "data" / "annotations"
TABLES = ROOT / "results" / "tables"
METRICS = ROOT / "results" / "metrics"
CORPUS = ROOT / "data" / "interim" / "corpus_v1_clean.jsonl"
TAXONOMY = ROOT / "configs" / "taxonomy.yaml"
SPLIT = ROOT / "configs" / "split.yaml"
pd.set_option("display.width", 200, "display.max_colwidth", 70, "display.max_rows", 200)


def cmd_progress(_):
    rows = []
    for who in gold.MEMBERS:
        for label, reader in (("pilot", gold.read_pilot), ("batch", gold.read_batch)):
            df = reader(who, ANN)
            rows.append({"member": who, "sheet": label, "labelled": int((df["has_event"] != "").sum()),
                         "total": len(df), "events_yes": int((df["has_event"] == "yes").sum())})
    print(pd.DataFrame(rows).to_string(index=False))


def cmd_validate(a):
    tax, corpus = gold.load_yaml(TAXONOMY), gold.load_corpus_text(CORPUS)
    if not corpus:
        print(f"(corpus not found at {CORPUS}; trigger check uses the 700-char excerpt only)")
    TABLES.mkdir(parents=True, exist_ok=True)
    whos = gold.MEMBERS if a.who.upper() == "ALL" else [a.who.upper()]
    for who in whos:
        df = gold.read_pilot(who, ANN) if a.pilot else gold.read_batch(who, ANN)
        total = len(df)
        df = df[df["has_event"] != ""] if a.partial else df
        label = "pilot" if a.pilot else "batch"
        issues = gold.validate(df, tax, corpus)
        out = TABLES / f"annotation_issues_{label}_{who}.csv"
        issues.to_csv(out, index=False)
        n_err = int((issues["severity"] == "ERROR").sum())
        n_warn = int((issues["severity"] == "WARN").sum())
        print(f"\n== {who} {label}: {len(df)}/{total} rows checked | {n_err} ERROR | {n_warn} WARN -> {out.name}")
        if len(issues):
            print(issues.sort_values(["severity", "annotation_id"]).to_string(index=False))


def cmd_adjudicate(_):
    rows, review = gold.adjudicate_pilot(ANN)
    TABLES.mkdir(parents=True, exist_ok=True)
    rows.to_csv(ANN / "pilot_adjudicated.csv", index=False, encoding="utf-8")
    review.to_csv(TABLES / "pilot_adjudication_review.csv", index=False, encoding="utf-8")
    print(review[["annotation_id", "final_has_event", "final_category", "tie_broken_by_HA"]].to_string(index=False))
    print(f"\nwrote data/annotations/pilot_adjudicated.csv ({len(rows)} rows) and "
          f"results/tables/pilot_adjudication_review.csv. Rows with tie_broken_by_HA=True need your own check.")


def cmd_validate_extra(_):
    tax, corpus = gold.load_yaml(TAXONOMY), gold.load_corpus_text(CORPUS)
    for p in sorted(ANN.glob("annotation_extra_*.xlsx")):
        df = gold.read_sheet(p, "Batch_EXTRA")
        df["annotator"] = p.stem.split("_")[-1]
        df = df[df["has_event"] != ""]
        issues = gold.validate(df, tax, corpus)
        print(f"\n== {p.name}: {len(df)} labelled rows | {(issues['severity'] == 'ERROR').sum()} ERROR | "
              f"{(issues['severity'] == 'WARN').sum()} WARN")
        if len(issues):
            print(issues.sort_values(["severity", "annotation_id"]).to_string(index=False))


def cmd_agreement(_):
    table, dis = gold.pilot_agreement(ANN)
    TABLES.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    table.round(3).to_csv(METRICS / "pilot_agreement.csv", index=False)
    dis.to_csv(TABLES / "pilot_disagreements.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(f"\n{len(dis)} disagreement rows -> results/tables/pilot_disagreements.csv (bring this to the meeting)")
    print("Bands: kappa >= 0.70 good | 0.60-0.70 acceptable | < 0.60 rewrite the rule that caused it")


def cmd_qa_make(a):
    d = gold.make_qa(ANN, n=a.n)
    print(f"QA sheets written to {d}. Send qa_<reviewer>.xlsx to each member (reviewer HA checks VV, VV checks SH, "
          f"SH checks TS, TS checks HA). Keep qa_key.csv to yourself until they return the sheets.")


def cmd_qa_report(_):
    summary, mism = gold.qa_report(ANN)
    if summary.empty:
        print("No filled QA sheets found in data/annotations/qa/ (expected qa_VV.xlsx etc.)")
        return
    TABLES.mkdir(parents=True, exist_ok=True)
    print(summary.to_string(index=False))
    mism.to_csv(TABLES / "qa_mismatches.csv", index=False)
    print(f"\n{len(mism)} mismatching rows -> results/tables/qa_mismatches.csv")


def cmd_target(_):
    cfg = splits.load_split(SPLIT)
    problems = splits.validate_split(cfg)
    print("split.yaml:", "consistent" if not problems else problems)
    print("locked:", cfg.get("locked"), "| cutoff:", cfg["cutoff_month"])
    for name, (s, e) in cfg["blocks"].items():
        print(f"  {name:<10} {s} -> {e}  ({len(splits.months(s, e))} months)")
    gp = ANN / "gold_events_v1.csv"
    if gp.exists():
        tax = gold.load_yaml(TAXONOMY)
        ev = pd.read_csv(gp, dtype=str, keep_default_na=False)
        df = splits.monthly_labels(ev, cfg, tax["category_precedence"])
        print("\nGOLD events only (just a sanity check; the real series comes from the event DB in Week 2):")
        print(df[df["n_events"] > 0].to_string(index=False))
    else:
        print("\n(no gold_events_v1.csv yet; run `merge` first to preview labels)")


def cmd_merge(a):
    cfg = splits.load_split(SPLIT)
    tax, corpus = gold.load_yaml(TAXONOMY), gold.load_corpus_text(CORPUS)
    METRICS.mkdir(parents=True, exist_ok=True)
    s = gold.merge_gold(ANN, tax, cfg, corpus, ANN / "gold_events_v1.csv", ANN / "gold_all_labels_v1.csv",
                        METRICS / "gold_summary_v1.json", allow_incomplete=a.allow_incomplete)
    for k, v in s.items():
        print(f"{k}: {v}")
    if not s["meets_minimum_200"]:
        print("\n!! Fewer than 200 events: label more documents (the plan's hard minimum is 200).")
    if s["categories_under_10_events"]:
        print("\n!! Categories with < 10 events (macro-F1 will be unstable):", s["categories_under_10_events"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--who", default="ALL", help="VV, SH, TS, HA or ALL (default)")
    v.add_argument("--pilot", action="store_true", help="check the Pilot sheet instead of the Batch sheet")
    v.add_argument("--partial", action="store_true", help="skip rows you have not labelled yet")
    v.set_defaults(fn=cmd_validate)
    sub.add_parser("progress").set_defaults(fn=cmd_progress)
    sub.add_parser("adjudicate-pilot").set_defaults(fn=cmd_adjudicate)
    sub.add_parser("validate-extra").set_defaults(fn=cmd_validate_extra)
    sub.add_parser("agreement").set_defaults(fn=cmd_agreement)
    q = sub.add_parser("qa-make")
    q.add_argument("--n", type=int, default=10)
    q.set_defaults(fn=cmd_qa_make)
    sub.add_parser("qa-report").set_defaults(fn=cmd_qa_report)
    sub.add_parser("target").set_defaults(fn=cmd_target)
    m = sub.add_parser("merge")
    m.add_argument("--allow-incomplete", action="store_true", help="draft merge while labelling is unfinished")
    m.set_defaults(fn=cmd_merge)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
