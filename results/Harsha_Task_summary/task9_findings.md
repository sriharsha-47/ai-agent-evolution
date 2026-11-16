# Task 9 (D9): relation extraction (organisation - system - capability - action)

Code: `src/ainevo/extraction/relations.py` (`extract_relation`), eval: `scripts/eval_relations.py`. Gold: gold_events_v1_1.csv (217 events).
Run: `python scripts/eval_relations.py data/interim/corpus_v1_clean.jsonl <gold_events_v1_1.csv>`

## Results
| part | method | result |
|---|---|---|
| organisation - system pair | D5 org rules + D5 system-name rules, first system only (lenient match) | P 0.72, R 0.50, F1 0.59 (gold 107 pairs, predicted 74) |
| capability (13 tags) | keyword rules | holdout acc 0.477 / macro-F1 0.387; all 217: 0.521 / 0.505 |
| capability | TF-IDF + LR | holdout 0.364 / 0.284; 5-fold CV 0.456 / 0.370 |
| capability | majority class | holdout 0.136 / 0.022 |
| action phrase (unigram F1 vs gold text) | trigger clause (extractive) | 0.232 (blogs 0.193, papers 0.287, release notes 0.069) |
| action phrase | title only | 0.311 (blogs 0.241, papers 0.410, release notes 0.015) |
| action phrase | first 22 words | 0.173 |

## Reading
1. Pairs: recall is limited by system names (found for about half of blogs and papers, D8) and by partner organisations (D5). Precision is fine.
2. Capability: hand-written keyword rules beat the learned model (0.477 vs 0.364 on 44 holdout rows), because 13 classes over 173 training rows is too few for TF-IDF. Caveat: I wrote the rules from the tag definitions after seeing the tag counts and a handful of gold rows, so the rule numbers are somewhat optimistic. Capability also depends strongly on category (e.g. multi_agent_coordination category -> capability), which a later stage could use, but it is not used here to avoid leakage.
3. Action: gold actions are free-text paraphrases; unigram F1 is a crude score, and the paper title beats the extractive clause for papers (0.410 vs 0.287). For release notes the trigger clause is poor (0.069): the gold actions are paraphrased feature summaries. A generative step (LLM summary, later) would be the real fix; not built.
4. Do not report action F1 as a headline number; report org-system pair F1 and capability macro-F1.
