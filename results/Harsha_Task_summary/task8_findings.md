# Task 8 (D8): full extraction pipeline on corpus_v1_clean (3,018 docs)

Run: `python scripts/run_extraction_pipeline.py data/interim/corpus_v1_clean.jsonl --out data/processed` (about 10 s, no spaCy, no network).
Outputs: `data/processed/extractions_v1.jsonl` (full, includes unit text, keep out of public repos), `extractions_v1.csv` (flat, no raw text), `extraction_summary.txt`.
Code: `src/ainevo/extraction/pipeline.py` (`extract_document`), chaining segment_events + guess_organizations + guess_systems + temporal.

## Coverage
| source_type | docs | trigger unit found | organization | system name |
|---|---|---|---|---|
| lab_blog | 1308 | 87% | 100% | 46% |
| paper | 984 | 90% | 0% (by design, guideline v1) | 47% |
| release_notes | 726 | 70% | 100% | 100% |

`has_candidate_event` only means a trigger unit exists. It is not a has_event classifier: 87-90% of docs pass, far above the real event rate.

## Decision: event_date = publication_date
Checked against gold_events_v1 (non-VV rows, 179 events): gold date equals publication_date in 178/179. Text-derived dates were worse (172/179; they pick up past dates like "in 2024-10" in release notes and blog bodies). So `event_date` is the publication date; the best in-text date is kept in `text_date` for inspection only.

## Trigger agreement with gold
Pipeline trigger overlaps the annotator's trigger (substring either way) in 98/179 (55%). Gold triggers are free-form copies chosen by hand, so this is a loose lower bound; the Task 2 top-1/top-3 numbers on the 90 docs remain the better measure.

## Known limits
- system name missing on about half of blogs and papers (no title name, no model in the opening).
- Only the first trigger unit per document is the primary; multi-event documents keep up to 5 units.
- Papers have no organization, as in gold v1.
