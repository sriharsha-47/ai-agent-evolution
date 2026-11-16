# Data dictionary and corpus statistics

Raw data, the cleaned corpus and the database are **not** stored in Git. This file says what each file is, how to
re-create it and what is in it. Full counts are in `docs/corpus_report.txt` (output of
`py -3.13 scripts/build_db.py --stats`).

## Folders

| folder | content | in Git? |
|---|---|---|
| `raw/` | untouched downloads: `arxiv/`, `lab_blogs/`, `news/`, `incidents/` | no |
| `interim/` | merged and cleaned corpus: `corpus_v1.jsonl`, `corpus_v1_clean.jsonl`, `cleaning_report.json` | no (share via the common drive) |
| `processed/` | `events.db` (SQLite), `extracted_events.csv`, `forecast_latest.json` | no |
| `annotations/` | gold labels and annotation sheets | yes (owner: Harshita) |
| `seed/` | seed timeline from Epoch AI / AI Index (owner: Saketh) | yes |

## Corpus v1 (frozen candidate)

File: `data/interim/corpus_v1_clean.jsonl`
SHA256: `48d3d61e15963f13a78a3b7e030d7882fa9b4a75e3ce3d4b78d2560208a88b09`

3,018 documents from 2017 to September 2026: papers 984, lab blog posts 1,308, release notes 726.

| year | lab_blog | paper | release_notes | total |
|------|----------|-------|---------------|-------|
| 2017 | 27 | 41 | 0 | 68 |
| 2018 | 18 | 42 | 0 | 60 |
| 2019 | 18 | 41 | 0 | 59 |
| 2020 | 16 | 42 | 0 | 58 |
| 2021 | 43 | 42 | 0 | 85 |
| 2022 | 92 | 82 | 0 | 174 |
| 2023 | 161 | 153 | 135 | 449 |
| 2024 | 225 | 192 | 176 | 593 |
| 2025 | 337 | 195 | 222 | 754 |
| 2026 | 371 | 154 | 193 | 718 |

Top sources: arXiv 984, OpenAI 613, Hugging Face 434, langchain-ai/langchain 341, crewAIInc/crewAI 175,
Anthropic 149, Significant-Gravitas/AutoGPT 115, microsoft/autogen 95, Google DeepMind 61, Google Research 51.

Cleaning caps per month: OpenAI 15, Hugging Face 8, release notes 8 per repository. The 315 documents annotated
in the pilot pool (`annotations/pool_v0.json`) are never dropped by cleaning.

Known gaps:
- 2017 to 2021 are below the planned 80 documents per year, and 2022 is far below the planned 350.
- No Meta, Mistral, DeepSeek or Qwen sources; Microsoft has only 10 documents.
- News and incident documents are collected separately (Saketh) and are not in this file.
- 29 documents are dated October 2026. They stay in the corpus but are outside the forecasting cutoff
  (`cutoff_month: 2026-09` in `configs/split.yaml`).

### Document fields

| field | meaning |
|---|---|
| `doc_id` | `<first two letters of source_type>_<first 10 characters of md5(url)>`, for example `la_c0eeb1f9ee` |
| `title` | document title |
| `publication_date` | date of publication (a missing date is never guessed) |
| `source` | arXiv, OpenAI, Anthropic, Hugging Face, a GitHub repository and so on |
| `source_type` | `paper`, `lab_blog` or `release_notes` |
| `url` | original link |
| `text` | full page text where available, otherwise abstract or snippet |
| `retrieval_date` | when it was downloaded |
| `query_used` | the query or collector rule that found it |
| `metadata` | JSON with flags: `llm_terms_found`, `low_signal_release`, `text_source` |

## Event database

Build it with `py -3.13 scripts/build_db.py --events data/annotations/gold_events_v1.csv`. It creates
`data/processed/events.db` with the documents (and a full-text index), the events and the links between them.
Add extracted events with `py -3.13 scripts/load_extracted.py data/processed/extracted_events.csv`.

Every event has an `origin`:

| origin | meaning | count |
|---|---|---|
| `gold` | hand-labelled by the team under the frozen guidelines (v1) | 217 |
| `extracted` | provisional model output (thresholds 0.75 for lab blogs and 0.85 for papers, chosen on 112 labelled documents); 538 lab-blog and 333 paper events; none from release notes. Categories are about 60% accurate on gold | 871 |
Gold events by category: multi_agent_coordination 57, research_milestone 39, agentic_development 30,
capability_breakthrough 28, infrastructure_ecosystem 28, adoption_scale 15, autonomous_action 11,
unexpected_behavior_incident 9. (Check against `build_db.py --stats`.)

The forecasting series is built from 216 gold events (one duplicate is merged); extracted events are not used for forecasting.

### Event fields

| field | meaning |
|---|---|
| `event_id` | unique identifier (gold: `GE-NNNN`, extracted: `EX-NNNNN`) |
| `date` | event date, `YYYY-MM-DD` or `YYYY-MM` |
| `category` | one of the 8 categories in `configs/taxonomy.yaml` |
| `trigger` | exact words in the document that signal the event |
| `organization` | organization or lab |
| `model_agent` | model, bot or agent involved |
| `capability` | one primary capability tag |
| `action` | observed or developed action |
| `autonomy_level` | 0 to 4, or NA (scale in `configs/taxonomy.yaml`) |
| `interaction` | human_model, human_agent, agent_tool, agent_agent, agent_environment or ecosystem |
| `impact` | consequence or evidence |
| `source` / `doc_id` | the document the event comes from |
| `confidence` | gold: 1 to 3 (annotator). Extracted: 0 to 1 (model probability) |

## Annotations (`data/annotations/`)

| file | content |
|---|---|
| `annotation_<initials>.xlsx` | each member's annotation sheet (Pilot and Batch) |
| `pool_v0.json`, `annotation_pool_v0.csv` | the 315 documents drawn for the pilot pool |
| `gold_events_v1.csv` | the 217 gold events (documents labelled `yes`) |
| `gold_all_labels_v1.csv` | all 395 labelled documents: 217 `yes`, 140 `no_not_significant`, 38 `no_off_topic` |
| `threshold_sample.xlsx` | 150 random documents labelled to choose the extraction threshold |

`has_event` values: `yes`, `no_not_significant`, `no_off_topic`, `no_unclear_text`. Rules are in
`docs/annotation_guidelines.md` and `configs/taxonomy.yaml`.

## Re-creating the data

1. `py -3.13 scripts/run_collection.py` collects arXiv papers, lab blogs and GitHub release notes into `data/raw/`.
   It is slow (lab blogs take 1 to 2 hours) and overwrites raw data. Set `$env:GITHUB_TOKEN` first to raise the
   GitHub rate limit. Collectors respect `robots.txt`, log `retrieval_date` and `query_used`, and keep source
   information.
2. `py -3.13 scripts/run_pipeline.py` merges, cleans (boilerplate, dates) and deduplicates (exact and MinHash
   near-duplicates) into `data/interim/corpus_v1_clean.jsonl` and `cleaning_report.json`.
3. `py -3.13 scripts/check_corpus.py` sanity-checks the result.
4. `py -3.13 scripts/build_db.py --events data/annotations/gold_events_v1.csv` builds the database.

Re-collecting later gives a different corpus (new documents and changed pages). Use the shared
`corpus_v1_clean.jsonl` and its SHA256 above to reproduce the reported results.

## Hygiene

Respect robots.txt, terms of service and rate limits. Keep raw news text out of any public repository. Do not
commit API keys, `.env`, database files or anything under `raw/`, `interim/` or `processed/`.
