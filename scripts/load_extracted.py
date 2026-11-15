"""Add Harsha's extracted events to the existing event database (origin='extracted'). Owner: VV.

Run from the repo root, after build_db.py has built the database with the gold events:
    py -3.13 scripts/load_extracted.py data/processed/extracted_events.csv
Safe to re-run (same event_id replaces the old row). Close Streamlit first.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ainevo.storage import db  # noqa: E402

DB_PATH = ROOT / "data" / "processed" / "events.db"


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: py -3.13 scripts/load_extracted.py <extracted_events.csv>")
    path = Path(sys.argv[1])
    if not path.exists():
        raise SystemExit(f"events file not found: {path}")
    if not DB_PATH.exists():
        raise SystemExit("database not found: run scripts/build_db.py --events <gold csv> first")
    conn = db.connect(DB_PATH)
    try:
        tax = db.load_taxonomy(str(ROOT / "configs" / "taxonomy.yaml"))
        n, rejected = db.load_events_file(conn, path, tax, origin="extracted")
        print(f"loaded {n} extracted events, rejected {len(rejected)}")
        for eid, problems in rejected[:10]:
            print("  rejected", eid, problems)
        for origin, cnt in conn.execute("SELECT origin, COUNT(*) FROM events GROUP BY origin ORDER BY origin"):
            print(f"  {origin}: {cnt}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
