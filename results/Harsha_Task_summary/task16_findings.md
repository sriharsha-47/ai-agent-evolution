# Task 16 (D16): robustness - per-source and per-period metrics

Run: `python scripts/eval_robustness.py data/interim/corpus_v1_clean.jsonl <gold_events_v1_1.csv> <gold_all_labels_v1.csv>`  ->  `results/tables/d16_robustness.csv` (all slices, `low_n` flags n < 15) and `d16_robustness_report.txt`.
Classification = out-of-fold TF-IDF + LR (5-fold, one seed). Extraction = rule extractors (D5/D8). Period = publication year (<=2021, 2022, 2023, 2024, 2025, 2026). Slices marked * have n < 15: anecdotes.

## Classification by source (217 gold events; has_event on 395 labelled docs)
| task | all | lab_blog | paper | release_notes |
|---|---|---|---|---|
| category accuracy / macro-F1 | 0.590 / 0.504 | 0.616 / 0.413 | 0.575 / 0.280 | 0.556 / 0.539 (n=18) |
| autonomy accuracy | 0.562 | 0.570 | 0.566 | 0.500 |
| interaction accuracy | 0.631 | 0.640 | 0.655 | 0.444 |
| has_event accuracy / F1(yes) | 0.820 / 0.848 | 0.778 / 0.857 | 0.829 / 0.904 | 0.853 / **0.095** |

- Accuracy looks similar across sources, but macro-F1 is much lower for papers (0.28): three classes (multi_agent, research_milestone, agentic_development) hold most papers and the rare classes are never found.
- has_event on release notes: 85% accuracy is only because 86% of them are negatives; F1 for "yes" is 0.095. The model cannot find release-note events (same finding as the threshold work).

## Classification by period (category accuracy / macro-F1)
<=2021: 0.455 / 0.484 (n=11*), 2022: 0.750 / 0.600 (n=4*), 2023: 0.593 / 0.518 (27), 2024: 0.523 / 0.316 (44), 2025: 0.552 / 0.506 (67), 2026: 0.688 / 0.564 (64).
No monotone trend; 2024 is the weakest well-populated year. Gold has only 15 events up to 2022, so early-period metrics cannot be trusted; use the quarterly view there, as the plan says.
has_event accuracy by period: <=2021 0.514 (n=37), 2022 0.462 (n=13*), 2023 0.727, 2024 0.897, 2025 0.944, 2026 0.843. The detector works for the recent LLM-agent period and is near chance on the early documents (different vocabulary, few events): do not use has_event predictions before 2023 without review.

## Extraction by source and period
| | all | lab_blog | paper | release_notes |
|---|---|---|---|---|
| organization F1 | 0.880 | 0.855 | n/a (gold blank) | 1.000 |
| model_agent F1 (P / R) | 0.674 (0.79 / 0.59) | 0.605 | 0.685 | 0.889 |

Period (organization F1 / model_agent F1): 2023 1.00 / 0.82, 2024 1.00 / 0.76, 2025 0.72 / 0.63, 2026 0.87 / 0.64, <=2021 0.96 / 0.50 (n=11*), 2022 1.00 / 0.33 (n=4*).
- Organization drops in 2025 (0.72): partner and customer stories, where the actor is not the source lab.
- model_agent is weaker in 2025-26 (0.63) than 2023-24 (0.76-0.82). The misses in 2025+ are mostly product, tool or programme names that no gazetteer can know (OpenEnv, CUGA, AprielGuard, ScreenEnv, GABRIEL, Safety Bug Bounty program) plus descriptive gold labels ("OpenAI models"), not model families. A "we introduce X" pattern would help; not built.

## Caveats
Small n per slice (most under 70); one CV seed; the extraction rules were written after reading these documents, so extraction numbers are optimistic; no confidence intervals. Report macro-F1 with caution, as the plan states.
