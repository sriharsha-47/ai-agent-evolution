"""Week-1 gold-set tooling (owner: HA).

Reads the four annotation workbooks (data/annotations/annotation_XX.xlsx) and provides:
  - validate():        check every label against taxonomy.yaml + sanity rules
  - pilot_agreement(): Fleiss / Cohen kappa on the shared 15-document pilot
  - make_qa()/qa_report(): blind cross-check of 10 labels from another member's batch
  - merge_gold():      merge the four batches into gold_events_v1.csv (+ classifier hold-out flag)
Run through scripts/week1_gold.py.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
MEMBERS = ["VV", "SH", "TS", "HA"]
META = ["annotation_id", "doc_id", "publication_date", "source", "source_type", "title", "url", "text_excerpt"]
FIELDS = ["has_event", "event_date", "category", "trigger", "organization", "model_agent", "capability",
          "action", "autonomy_level", "interaction", "impact", "phase_hint", "confidence", "notes"]
DATE_RE = re.compile(r"^\d{4}-\d{2}(-\d{2})?$")
# reviewer -> author whose batch the reviewer checks (nobody checks their own)
QA_CYCLE = {"HA": "VV", "VV": "SH", "SH": "TS", "TS": "HA"}


# ----------------------------------------------------------------------------- loading
def load_yaml(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _clean(v) -> str:
    """Excel cell -> tidy string ('' for blank, '1.0' -> '1', '2024-05-01 00:00:00' -> '2024-05-01')."""
    s = "" if v is None else str(v).strip()
    if re.fullmatch(r"\d+\.0", s):
        s = s[:-2]
    m = re.fullmatch(r"(\d{4}-\d{2}-\d{2}) 00:00:00", s)
    if m:
        s = m.group(1)
    return s


def read_sheet(xlsx, sheet: str) -> pd.DataFrame:
    # keep_default_na=False so the autonomy label "NA" is NOT turned into NaN
    df = pd.read_excel(xlsx, sheet_name=sheet, dtype=str, keep_default_na=False)
    missing = [c for c in META + FIELDS if c not in df.columns]
    if missing:
        raise ValueError(f"{xlsx} / {sheet}: missing columns {missing} (do not rename columns)")
    return df[META + FIELDS].map(_clean)


def read_batch(who: str, folder: Path) -> pd.DataFrame:
    df = read_sheet(Path(folder) / f"annotation_{who}.xlsx", f"Batch_{who}")
    df["annotator"] = who
    return df


def read_pilot(who: str, folder: Path) -> pd.DataFrame:
    df = read_sheet(Path(folder) / f"annotation_{who}.xlsx", "Pilot")
    df["annotator"] = who
    return df


def load_corpus_text(path) -> dict:
    """doc_id -> full text (empty dict if the corpus file is not there)."""
    p = Path(path)
    out = {}
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            out[r["doc_id"]] = r["text"]
    return out


# ----------------------------------------------------------------------------- validation
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def validate(df: pd.DataFrame, tax: dict, corpus: dict | None = None) -> pd.DataFrame:
    """Return one row per problem: annotation_id, annotator, field, severity (ERROR/WARN), message."""
    corpus = corpus or {}
    cats = [c["id"] for c in tax["categories"]]
    default_phase = {c["id"]: c["default_phase"] for c in tax["categories"]}
    allowed = {
        "has_event": list(tax["has_event_values"]),
        "category": cats,
        "capability": list(tax["capability_tags"]),
        "autonomy_level": ["0", "1", "2", "3", "4", "NA"],
        "interaction": list(tax["interaction"]),
        "phase_hint": [str(i) for i in range(1, 8)],
        "confidence": ["1", "2", "3"],
    }
    issues = []

    def add(r, field, sev, msg):
        issues.append({"annotation_id": r["annotation_id"], "annotator": r.get("annotator", ""),
                       "field": field, "severity": sev, "message": msg})

    for _, r in df.iterrows():
        he = r["has_event"]
        if he not in allowed["has_event"]:
            add(r, "has_event", "ERROR", f"missing or invalid value {he!r}")
            continue
        filled = [f for f in FIELDS if f not in ("has_event", "notes") and r[f]]
        if he != "yes":
            if filled:
                add(r, "fields", "WARN", "has_event is not 'yes' but these fields are filled: " + ", ".join(filled))
            continue

        for f in ("event_date", "category", "trigger", "capability", "autonomy_level",
                  "interaction", "phase_hint", "confidence"):
            if not r[f]:
                add(r, f, "ERROR", "required when has_event = yes")
        for f, vals in allowed.items():
            if f != "has_event" and r[f] and r[f] not in vals:
                add(r, f, "ERROR", f"{r[f]!r} is not an allowed value")

        d = r["event_date"]
        if d:
            if not DATE_RE.match(d):
                add(r, "event_date", "ERROR", f"{d!r} is not YYYY-MM-DD or YYYY-MM")
            else:
                if d[:4] < "2017":
                    add(r, "event_date", "WARN", "before 2017 (outside the project range)")
                if r["publication_date"] and d[:7] > r["publication_date"][:7]:
                    add(r, "event_date", "WARN", "event_date is later than publication_date")

        if r["trigger"]:
            text = corpus.get(r["doc_id"]) or r["text_excerpt"]
            if _norm(r["trigger"]) not in _norm(text):
                add(r, "trigger", "WARN", "trigger text not found in the document (copy the exact words)")

        cat = r["category"]
        if cat in default_phase and r["phase_hint"].isdigit():
            if int(r["phase_hint"]) not in default_phase[cat]:
                add(r, "phase_hint", "WARN", "differs from the category's default phase (fine if the text shows it)")
        if cat == "autonomous_action" and r["autonomy_level"] in ("0", "1", "2"):
            add(r, "autonomy_level", "WARN", "autonomous_action usually means autonomy 3-4")
        if cat == "multi_agent_coordination" and r["interaction"] and r["interaction"] != "agent_agent":
            add(r, "interaction", "WARN", "multi_agent_coordination usually has interaction = agent_agent")
        if r["confidence"] == "1" and not r["notes"]:
            add(r, "notes", "WARN", "confidence 1 but no note explaining the doubt")

    return pd.DataFrame(issues, columns=["annotation_id", "annotator", "field", "severity", "message"])


# ----------------------------------------------------------------------------- pilot agreement
def pilot_agreement(folder: Path):
    """Returns (agreement table, disagreement table) for the 4 pilot sheets."""
    from ainevo.evaluation.agreement import compare, disagreements

    frames = {}
    for who in MEMBERS:
        df = read_pilot(who, folder)
        if (df["has_event"] == "").any():
            raise SystemExit(f"{who}'s pilot is not fully labelled yet "
                             f"({(df['has_event'] == '').sum()} blank has_event cells).")
        df["has_event_binary"] = df["has_event"].map(lambda x: "yes" if x == "yes" else "no")
        for f in ("category", "autonomy_level", "interaction"):
            df[f] = df[f].replace("", "(none)")
        frames[who] = df

    fields = ("has_event", "has_event_binary", "category", "autonomy_level")
    all_docs = compare(frames, fields=fields)
    all_docs["scope"] = "all pilot docs"

    # category / autonomy / interaction only among docs that ALL four called an event
    yes_ids = set.intersection(*(set(f.loc[f["has_event"] == "yes", "doc_id"]) for f in frames.values()))
    parts = [all_docs]
    if len(yes_ids) >= 2:
        sub = {w: f[f["doc_id"].isin(yes_ids)] for w, f in frames.items()}
        ev = compare(sub, fields=("category", "autonomy_level", "interaction"))
        ev["scope"] = "docs all four called an event"
        parts.append(ev)
    table = pd.concat(parts, ignore_index=True)[["scope", "field", "pair", "kappa", "n", "verdict"]]

    titles = frames["HA"].set_index("doc_id")["title"].to_dict()
    dis = []
    for f in fields + ("interaction",):
        d = disagreements(frames, f)
        if len(d):
            d = d.copy()
            d.insert(0, "field", f)
            d.insert(1, "title", [titles.get(i, "") for i in d.index])
            dis.append(d.reset_index())
    dis = pd.concat(dis, ignore_index=True) if dis else pd.DataFrame()
    return table, dis


# ----------------------------------------------------------------------------- pilot adjudication (no meeting)
def _vote(values: list, prefer: str, order: list | None = None):
    """Majority value. Returns (value, was_tie). On a tie: `prefer` if it is among the leaders, else first leader."""
    c = Counter(values)
    top = max(c.values())
    leaders = [v for v in (order or list(c)) if c.get(v, 0) == top]
    if len(leaders) == 1:
        return leaders[0], False
    return (prefer if prefer in leaders else leaders[0]), True


def adjudicate_pilot(folder: Path):
    """Majority vote over the four pilot sheets (ties -> HA's label). Returns (gold rows, review table)."""
    frames = {w: read_pilot(w, folder) for w in MEMBERS}
    for w, f in frames.items():
        if (f["has_event"] == "").any():
            raise SystemExit(f"{w}'s pilot is not fully labelled yet.")
    rows, review = [], []
    for doc in list(frames["HA"]["doc_id"]):
        recs = {w: f[f["doc_id"] == doc].iloc[0] for w, f in frames.items()}
        ha = recs["HA"]
        he, tie_he = _vote([r["has_event"] for r in recs.values()], ha["has_event"])
        out = {c: ha[c] for c in META}
        out.update({f: "" for f in FIELDS})
        out["has_event"] = he
        tie_cat, votes_cat = False, ""
        if he == "yes":
            yes = {w: r for w, r in recs.items() if r["has_event"] == "yes" and r["category"]}
            cat, tie_cat = _vote([r["category"] for r in yes.values()], ha["category"])
            votes_cat = dict(Counter(r["category"] for r in yes.values()))
            same = {w: r for w, r in yes.items() if r["category"] == cat}
            src_w = next(w for w in ["HA", "VV", "SH", "TS"] if w in same)   # who supplies the free-text fields
            src = same[src_w]
            for f in FIELDS:
                out[f] = src[f]
            out["category"] = cat
            for f in ("capability", "autonomy_level", "interaction", "phase_hint"):
                out[f], _ = _vote([r[f] for r in same.values()], src[f])
            out["notes"] = f"adjudicated by majority vote; text fields from {src_w}"
        else:
            out["notes"] = "adjudicated by majority vote"
        rows.append(out)
        review.append({"annotation_id": ha["annotation_id"], "doc_id": doc, "title": ha["title"],
                       "final_has_event": he, "has_event_votes": dict(Counter(r["has_event"] for r in recs.values())),
                       "final_category": out["category"], "category_votes": votes_cat,
                       "tie_broken_by_HA": bool(tie_he or tie_cat)})
    return pd.DataFrame(rows, columns=META + FIELDS), pd.DataFrame(review)


# ----------------------------------------------------------------------------- QA cross-check
def make_qa(folder: Path, n: int = 10, seed: int = 42):
    """Blind QA sheets: each reviewer relabels n docs from another member's batch (see QA_CYCLE)."""
    qa_dir = Path(folder) / "qa"
    short = {a: int((read_batch(a, folder)["has_event"] != "").sum()) for a in QA_CYCLE.values()}
    short = {a: k for a, k in short.items() if k < n}
    if short:
        raise SystemExit(f"Cannot build QA sheets yet: these batches have fewer than {n} labelled rows "
                         f"{short}. Run qa-make after the batches are labelled.")
    qa_dir.mkdir(parents=True, exist_ok=True)
    key_rows = []
    for reviewer, author in QA_CYCLE.items():
        df = read_batch(author, folder)
        df = df[df["has_event"] != ""]
        yes, no = df[df["has_event"] == "yes"], df[df["has_event"] != "yes"]
        k_no = min(len(no), n // 2)
        k_yes = min(len(yes), n - k_no)
        k_no = min(len(no), n - k_yes)
        rng = random.Random(f"{seed}-{author}")
        picked = rng.sample(sorted(yes["annotation_id"]), k_yes) + rng.sample(sorted(no["annotation_id"]), k_no)
        sample = df[df["annotation_id"].isin(picked)].sort_values("annotation_id")
        sheet = sample[META].copy()
        for c in ("has_event", "category", "autonomy_level", "notes"):
            sheet[c] = ""
        sheet.to_excel(qa_dir / f"qa_{reviewer}.xlsx", sheet_name="QA", index=False)
        for _, r in sample.iterrows():
            key_rows.append({"reviewer": reviewer, "author": author, "annotation_id": r["annotation_id"],
                             "doc_id": r["doc_id"], "has_event": r["has_event"],
                             "category": r["category"], "autonomy_level": r["autonomy_level"]})
    pd.DataFrame(key_rows).to_csv(qa_dir / "qa_key.csv", index=False)
    return qa_dir


def qa_report(folder: Path):
    """Compare each returned qa_XX.xlsx with the key. Returns (summary table, mismatch table)."""
    qa_dir = Path(folder) / "qa"
    kp = qa_dir / "qa_key.csv"
    if not kp.exists() or kp.stat().st_size == 0:
        raise SystemExit("qa_key.csv is missing or empty. Delete data/annotations/qa and run qa-make "
                         "again once the batches are labelled.")
    key = pd.read_csv(kp, dtype=str, keep_default_na=False)
    summary, mism = [], []
    for reviewer in QA_CYCLE:
        p = qa_dir / f"qa_{reviewer}.xlsx"
        if not p.exists():
            continue
        rev = read_sheet_loose(p, "QA")
        if (rev["has_event"] == "").all():      # sheet not returned / not filled yet
            continue
        rev = rev[["doc_id", "has_event", "category", "autonomy_level", "notes"]]
        k = key[key["reviewer"] == reviewer].merge(rev, on="doc_id", suffixes=("_orig", "_rev"))
        if k.empty:
            continue
        agree_he = (k["has_event_orig"] == k["has_event_rev"])
        both_yes = k[(k["has_event_orig"] == "yes") & (k["has_event_rev"] == "yes")]
        summary.append({
            "reviewer": reviewer, "author": QA_CYCLE[reviewer], "n": len(k),
            "has_event_agree": round(agree_he.mean(), 2),
            "category_agree_on_events": round((both_yes["category_orig"] == both_yes["category_rev"]).mean(), 2)
            if len(both_yes) else None,
            "autonomy_agree_on_events": round((both_yes["autonomy_level_orig"] == both_yes["autonomy_level_rev"]).mean(), 2)
            if len(both_yes) else None,
        })
        bad = k[(k["has_event_orig"] != k["has_event_rev"]) |
                ((k["has_event_orig"] == "yes") & ((k["category_orig"] != k["category_rev"]) |
                                                   (k["autonomy_level_orig"] != k["autonomy_level_rev"])))]
        mism.append(bad[["reviewer", "author", "annotation_id", "doc_id", "has_event_orig", "has_event_rev",
                         "category_orig", "category_rev", "autonomy_level_orig", "autonomy_level_rev", "notes"]])
    return pd.DataFrame(summary), (pd.concat(mism, ignore_index=True) if mism else pd.DataFrame())


def read_sheet_loose(xlsx, sheet: str) -> pd.DataFrame:
    df = pd.read_excel(xlsx, sheet_name=sheet, dtype=str, keep_default_na=False)
    return df.map(_clean)


# ----------------------------------------------------------------------------- merge -> gold set
def assign_holdout(events: pd.DataFrame, frac: float, seed: int) -> pd.DataFrame:
    """Random hold-out per category (classifier evaluation only; the forecast split is chronological)."""
    ev = events.reset_index(drop=True).copy()
    ev["classifier_split"] = "train"
    for cat, grp in ev.groupby("category"):
        ids = sorted(grp.index)
        random.Random(f"{seed}-{cat}").shuffle(ids)
        k = round(len(ids) * frac)
        if k == 0 and len(ids) >= 5:
            k = 1
        ev.loc[ids[:k], "classifier_split"] = "holdout"
    return ev


def merge_gold(folder: Path, tax: dict, split_cfg: dict, corpus: dict,
               out_events: Path, out_all: Path, out_summary: Path, allow_incomplete: bool = False) -> dict:
    folder = Path(folder)
    df = pd.concat([read_batch(w, folder) for w in MEMBERS], ignore_index=True)
    for p in sorted(folder.glob("annotation_extra_*.xlsx")):      # optional extra batch(es), sheet Batch_EXTRA
        ex = read_sheet(p, "Batch_EXTRA")
        ex["annotator"] = p.stem.split("_")[-1]
        df = pd.concat([df, ex], ignore_index=True)

    adj = folder / "pilot_adjudicated.csv"   # optional: pilot labels agreed in the meeting
    if adj.exists():
        a = pd.read_csv(adj, dtype=str, keep_default_na=False).map(_clean)
        a = a.reindex(columns=list(df.columns), fill_value="")
        a["annotator"] = "consensus"
        df = pd.concat([df, a], ignore_index=True)

    blank = df[df["has_event"] == ""]
    if len(blank) and not allow_incomplete:
        raise SystemExit(f"{len(blank)} rows have no has_event label yet "
                         f"({blank['annotator'].value_counts().to_dict()}). Finish them or use --allow-incomplete.")
    df = df[df["has_event"] != ""]

    dup = df[df.duplicated("doc_id", keep=False)]
    if len(dup):
        print(f"WARNING: {dup['doc_id'].nunique()} doc_ids appear more than once; keeping the first.")
        df = df.drop_duplicates("doc_id", keep="first")

    issues = validate(df, tax, corpus)
    errors = issues[issues["severity"] == "ERROR"]
    if len(errors) and not allow_incomplete:
        raise SystemExit(f"{len(errors)} ERROR(s) in labels; run `validate` and fix them first.")

    df.to_csv(out_all, index=False, encoding="utf-8")   # every labelled doc incl. no_* (scope examples)

    ev = df[df["has_event"] == "yes"].copy()
    ev = ev.sort_values(["event_date", "doc_id"]).reset_index(drop=True)
    ev.insert(0, "event_id", [f"GE-{i:04d}" for i in range(1, len(ev) + 1)])
    ev = ev.rename(columns={"event_date": "date", "url": "source_url", "confidence": "annotator_confidence"})
    ev = assign_holdout(ev, split_cfg["classifier_holdout"]["gold_fraction"], split_cfg["classifier_holdout"]["random_state"])
    keep = ["event_id", "doc_id", "annotator", "annotation_id", "date", "publication_date", "category", "trigger",
            "organization", "model_agent", "capability", "action", "autonomy_level", "interaction", "impact",
            "phase_hint", "annotator_confidence", "source", "source_type", "source_url", "title", "notes",
            "classifier_split"]
    ev[keep].to_csv(out_events, index=False, encoding="utf-8")

    cats = Counter(ev["category"])
    summary = {
        "labelled_docs": int(len(df)),
        "has_event_counts": df["has_event"].value_counts().to_dict(),
        "n_events": int(len(ev)),
        "meets_minimum_200": bool(len(ev) >= 200),
        "meets_target_300": bool(len(ev) >= 300),
        "events_by_category": dict(cats),
        "categories_under_10_events": sorted(c for c in [x["id"] for x in tax["categories"]] if cats.get(c, 0) < 10),
        "events_by_year": ev["date"].str[:4].value_counts().sort_index().to_dict(),
        "events_by_source_type": ev["source_type"].value_counts().to_dict(),
        "events_by_annotator": ev["annotator"].value_counts().to_dict(),
        "classifier_split": ev["classifier_split"].value_counts().to_dict(),
        "warnings_in_labels": int((issues["severity"] == "WARN").sum()),
    }
    Path(out_summary).parent.mkdir(parents=True, exist_ok=True)
    with open(out_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary
