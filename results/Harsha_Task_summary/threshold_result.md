# has_event threshold: result from the fresh labelled sample

Sample: 112 random non-gold documents (subset of threshold_sample.xlsx; 37 blog, 38 paper, 37 release notes), labelled by SH (38), TS (37), VV (37), has_event only, guideline v1. 51 yes / 52 no_not_significant / 9 no_off_topic. Models fitted on gold only; no overlap with gold.

## Thresholds (minimum sample precision 0.85, highest recall at that precision)
| source | threshold | precision | recall | F1 | events in sample |
|---|---|---|---|---|---|
| lab_blog | 0.75 | 0.85 | 0.74 | 0.79 | 23/37 |
| paper | 0.85 | 0.88 | 0.70 | 0.78 | 20/38 |
| release_notes | none (set to 1.01) | - | - | - | 8/37 |

Stability: without TS's labels (75 rows) lab_blog becomes 0.65 and paper stays 0.85. Treat the blog threshold as 0.65-0.75.

## Release notes: not exportable with this model
The has_event model ranks release-note events barely above chance (AUC 0.59, highest p_event among the 8 events 0.21), so no threshold gives any precision. A release-note-only model (AUC 0.66), the trigger score (0.69) and a feature-minus-fix word heuristic (0.78) are all weak on 8 positives. The extracted file therefore has no release-note events; gold already contains release-note events (18). Fixing this needs more release-note labels and a dedicated model.

## Labeller differences (important)
Yes-rates differ a lot between labellers: SH 0.58, VV 0.54, TS 0.24 (lab blogs: SH 0.92, VV 0.67, TS 0.25; release notes: SH 0.15, VV 0.50, TS 0.00). The three samples do not overlap, so some of this is sampling (12-13 documents per source each), but a gap of 0.92 vs 0.25 on blogs suggests TS applies "no_not_significant" much more strictly. The thresholds above depend on all three. A shared check set (about 20 documents labelled by all three) would measure agreement; the earlier 10% re-check was for gold, not for this sheet.

## Final file
`data/processed/extracted_events.csv`: 871 events (538 blog, 333 paper, 0 release notes); gold documents excluded; confidence = p_event; impact empty; origin = extracted. The earlier provisional file had 834 (364 blog, 470 paper).
Reproduce: `choose_threshold.py` on `data/annotations/threshold_sample_labelled.xlsx`, set release_notes to 1.01 in thresholds.json, then `export_extracted_events.py --thresholds data/processed/thresholds.json`.
