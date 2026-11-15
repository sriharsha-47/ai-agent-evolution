# ai-agent-evolution

NLP-based extraction of "AI canon events" from AI news, lab posts and papers, an evolution timeline of AI-agent
development, and a forecast of the dominant event category in the next month. Course project, 4 members.

Results describe **reported** AI developments in our corpus, with uncertainty and source links. They are not
predictions of a future "AI revolution".

Further reading: `docs/annotation_guidelines.md` (labelling rules), `docs/decisions.md` (team decisions),
`docs/demo_script.md` (how to present the app), `data/README.md` (data dictionary and corpus statistics).

## What is in the repository

| folder | content |
|---|---|
| `configs/` | `sources.yaml`, `taxonomy.yaml` (8 categories, 8 phases), `split.yaml` (forecast split and data cutoff) |
| `src/ainevo/` | collection, preprocessing, storage (SQLite), extraction, normalization, features, analysis, forecasting, evaluation |
| `scripts/` | command-line entry points (see "Run") |
| `app/` | Streamlit app: `streamlit_app.py` plus `pages/` and `components/` |
| `data/` | `raw/` and `processed/` are git-ignored; `annotations/` holds the gold labels |
| `results/` | metric tables and figures |
| `tests/` | unit tests |

## Install

Python 3.13 is used by the team. The commands below are for Windows PowerShell; on macOS or Linux use `python3`
and `export PYTHONPATH=src`.

```
py -3.13 -m pip install -r requirements.txt
```

Optional: copy `.env.example` to `.env` and add a `GITHUB_TOKEN` to raise the GitHub API rate limit for the
release-note collector. In PowerShell set it for the session with `$env:GITHUB_TOKEN="..."`.

## Data you need

Raw data and the database are not stored in Git. To run the app you need two things:

1. **The cleaned corpus**, `data/interim/corpus_v1_clean.jsonl` (3,018 documents, 2017 to September 2026).
   Get it from the team's shared folder. Its sha256 must be
   `48d3d61e15963f13a78a3b7e030d7882fa9b4a75e3ce3d4b78d2560208a88b09`. Check it with:
   ```
   Get-FileHash data\interim\corpus_v1_clean.jsonl -Algorithm SHA256
   ```
   (Alternatively rebuild it with `scripts/run_collection.py` and `scripts/run_pipeline.py`; collection needs
   internet and the lab blogs take 1 to 2 hours.)
2. **The gold events**, `data/annotations/gold_events_v1.csv` (217 hand-labelled events).

Optional inputs the app picks up when present:

| file | produced by | used by |
|---|---|---|
| `data/processed/extracted_events.csv` | extraction pipeline (provisional, threshold 0.8) | Timeline, origin "extracted" |
| `results/tables/features_monthly_gold.csv`, `counts_quarterly_gold.csv` | feature engineering | Phases page |
| `results/tables/phases.csv` | change-point analysis | phase bands on the Phases page |
| `data/processed/forecast_latest.json` | forecasting | Forecast and Evidence pages |

## Run

From the repository root:

```
py -3.13 scripts/build_db.py --events data/annotations/gold_events_v1.csv
py -3.13 scripts/load_extracted.py data/processed/extracted_events.csv     # optional
py -3.13 -m streamlit run app/streamlit_app.py
```

Stop Streamlit before rebuilding the database, otherwise Windows keeps `events.db` locked.

| command | what it does |
|---|---|
| `py -3.13 scripts/run_collection.py` | collect arXiv, lab blogs and GitHub releases into `data/raw/` (slow, overwrites raw data) |
| `py -3.13 scripts/run_pipeline.py` | merge, clean and deduplicate into `data/interim/corpus_v1_clean.jsonl` and `cleaning_report.json` |
| `py -3.13 scripts/check_corpus.py` | sanity-check the corpus |
| `py -3.13 scripts/build_db.py --events <gold csv>` | build `data/processed/events.db` (documents, gold events, full-text search) |
| `py -3.13 scripts/build_db.py --stats` | print corpus and event counts per year, source and category |
| `py -3.13 scripts/load_extracted.py <csv>` | add extracted events to the database (origin `extracted`) |
| `$env:PYTHONPATH="src"; py -3.13 -m pytest -q tests` | run the unit tests |

`make` targets exist for the same steps on systems with `make`.

## The app

| page | shows |
|---|---|
| Home | corpus size, events in the database, documents per year and source type |
| 1 Corpus | full-text search over 3,018 documents with source, type and year filters |
| 2 Timeline | events by category over time; filters for category, organization, capability, autonomy, origin and months; event details with source links |
| 3 Phases | monthly feature trends, dominant category per month, 2017-2021 quarterly view, phase bands when `phases.csv` exists |
| 4 Forecast | forecast of the next month's dominant category from `forecast_latest.json`, baselines, history and supporting events. Without that file it shows a clearly marked most-frequent placeholder |
| 5 Evidence | one event with its fields, source documents and the trigger in context |

The Timeline opens on **gold** events (hand-labelled). The 834 **extracted** events are provisional model output
and can be added with the Origin filter.

## Notes on the data

- Collectors respect `robots.txt` where they fetch web pages, log `retrieval_date` and `query_used`, and never
  guess a missing date.
- Corpus caps per month: OpenAI 15, Hugging Face 8, release notes 8 per repository.
- Known gaps: 2017 to 2021 and 2022 are below the planned document counts; no Meta, Mistral, DeepSeek or Qwen
  sources; Microsoft has only 10 documents. News and incident sources are collected separately.
- The forecast uses monthly series from January 2022 to the cutoff in `configs/split.yaml`, with a chronological
  train, validation and test split. Only about half of the months have enough events to carry a label.

## Team workflow

Branches are named `feat/<initials>-<task>`, `main` is protected, and changes go in through pull requests with
one review. Do not commit `data/raw`, `data/processed`, `.env` or database files.
