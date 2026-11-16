# has_event threshold and extracted-events export (request from Varun)

## Why a fresh sample is needed
The gold pool was not drawn at random: make_annotation_pool.py drops short docs, low-signal releases and post-2023 papers without LLM terms, and prefers documents with rare-category hints. So the gold yes-rate (55%) is not the corpus yes-rate, and every gold document was used to fit the has_event model, so no threshold can be tuned on it honestly. Out-of-fold precision on gold is only a rough guide.

## Out-of-fold (gold, 395 docs) precision / recall of "event" by threshold, and share of unseen docs predicted "event"
| threshold | precision | recall | unseen blogs | unseen papers | unseen release notes |
|---|---|---|---|---|---|
| 0.5 | 0.79 | 0.91 | 92% | 98% | 1% |
| 0.7 | 0.89 | 0.81 | 59% | 79% | 0% |
| 0.8 | 0.95 | 0.62 | 31% | 56% | 0% |
| 0.9 | 0.97 | 0.17 | 5% | 16% | 0% |

Release notes: only 18 of 129 labelled release notes are events, so the model almost never predicts "event" for them at any threshold. A release-note-specific threshold needs fresh labels.

## Provisional file (now)
`extracted_events_provisional.csv`: threshold 0.8 for every source, gold documents excluded, 834 events (364 blog, 470 paper, 0 release notes). confidence = p_event. impact is empty (not extracted). Treat as provisional, load with origin=extracted, keep gold as the default view.
Known weaknesses: 0 release-note events; model_agent empty for 350 rows and often "ChatGPT" for o1/other posts (the o1 rule fails on the non-breaking hyphen in "o1-preview"); trigger empty for 62 rows; organization empty for papers by design (469 rows); action is a 22-word extract of the trigger sentence, not a paraphrase.

## Final threshold (needs about 1-2 h of labelling)
1. `python scripts/make_threshold_sample.py data/interim/corpus_v1_clean.jsonl <gold_all_labels_v1.csv>` writes data/annotations/threshold_sample.xlsx: 150 random non-gold docs (50 per source type, not filtered toward events).
2. A human labels column H (has_event only: yes / no_not_significant / no_off_topic / no_unclear_text) under guideline v1. 150 rows is about 1-2 hours; it can be split over people (ask Harshita how).
3. `python scripts/choose_threshold.py <corpus> <gold_events_v1_1.csv> <gold_all_labels_v1.csv> <labelled_sample.xlsx>` picks per-source thresholds (lowest threshold with sample precision >= 0.85) and writes data/processed/thresholds.json.
4. `python scripts/export_extracted_events.py <corpus> <gold_events_v1_1.csv> <gold_all_labels_v1.csv> --thresholds data/processed/thresholds.json` writes the final data/processed/extracted_events.csv.
Each source has 50 labels, so a per-source threshold is rough (a source with fewer than 5 events falls back to the global choice). Report it with that caveat. The scripts were smoke-tested with random labels only; no threshold has been chosen yet.
