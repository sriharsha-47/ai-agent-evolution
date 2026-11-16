# Task 12 (D12): autonomy / interaction attributes and final extraction run

Code: `src/ainevo/classification/attributes.py` (keyword rules), `scripts/eval_attributes.py`, `scripts/run_final_extraction.py`.
Gold: gold_events_v1_1.csv, 217 events; split 173 train / 44 holdout; CV = 5-fold.

## Attribute models (accuracy / macro-F1)
| attribute | majority | keyword rules (holdout) | TF-IDF + LR (holdout) | TF-IDF + LR (CV, all 217) |
|---|---|---|---|---|
| autonomy_level (0,1,2,3,4,NA) | 0.318 / 0.097 | 0.273 / 0.204 | 0.545 / 0.506 | 0.562 / 0.398 |
| interaction (6 types) | 0.273 / 0.071 | 0.477 / 0.446 | 0.682 / 0.520 | 0.631 / 0.512 |

- Learned models beat my keyword rules for both attributes; the rules (written from the guideline definitions, not tuned) are not worth using beyond a sanity baseline.
- Autonomy is an ordinal scale: where gold and prediction are both non-NA, 95% of CV predictions are within one level. Exact levels are weaker: level 1 (recall 0.07, 15 rows) and level 4 (2 rows, never found) are essentially not learned; NA is hard (F1 0.37) because it is a "cannot judge" label.
- Interaction: agent_agent (F1 0.79) and human_model (0.82) are good; agent_tool (recall 0.20), human_agent (0.19) and ecosystem (0.40) are weak. These are labels annotators themselves vary on, so expect a ceiling.
- Small data (44 holdout rows): quote the CV column beside the holdout.

## Final run (`scripts/run_final_extraction.py`) over corpus_v1_clean (3,018 docs)
Output `data/processed/extractions_final.csv` (flat, no raw text): pipeline fields (organisations, systems, event_date = publication date) + `p_event`, `has_event_pred`, `category_pred`, `capability_pred` (rules), `autonomy_pred`, `interaction_pred`, `action`, `in_gold`. Attributes are filled only where has_event_pred = yes. A 50-row sample is in `results/tables/extractions_final_sample50.csv`.

| source | docs | predicted events |
|---|---|---|
| lab_blog | 1308 | 1170 (89%) |
| paper | 984 | 937 (95%) |
| release_notes | 726 | 22 (3%) |

## Warning: has_event_pred is not reliable enough for the timeline
On documents the model has not seen, it predicts "event" for 92% of blogs and 98% of papers, while only 55% of the 395 labelled docs are events (blogs 68%, papers 81%), and for release notes it predicts 1% against 14% in the labelled sample. The binary classifier (CV accuracy 0.82 on the labelled docs) was trained on a sample that is not representative of the full corpus, and its probabilities are not calibrated. Before using it to build the timeline: choose the threshold on a fresh random labelled sample, or use a ranking (`p_event`) and review the top/bottom, and re-fit when the next annotation batch arrives. Category counts above are therefore over-counts.

## Not done
Embedding-based attribute models (the 395-doc embeddings exist, the 3,018-doc ones need another Colab run) and a re-run after the corpus freeze (Varun, D11). Re-run the script, do not edit the CSV.
