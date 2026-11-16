# Task 11 (D11): TF-IDF + logistic regression baseline on gold v1.1

Data: `gold_events_v1_1.csv` (217 events, VV rows re-labelled, no placeholders, all triggers verbatim in corpus text) and `gold_all_labels_v1.csv` (395 docs incl. 177 negatives).
Input text = source_type + title + first 3000 chars of the document. No gold column is used as a feature.
Run: `python scripts/train_baseline_classifier.py <corpus.jsonl> <gold_events_v1_1.csv> <gold_all_labels_v1.csv>`

## Category (8 classes), given split: train 173 / holdout 44
| model | accuracy | macro-F1 |
|---|---|---|
| majority class | 0.250 | 0.050 |
| source_type only | 0.409 | 0.174 |
| **TF-IDF + LR** | **0.568** | **0.477** |
| TF-IDF + LR, 5-fold CV on all 217 | 0.590 | 0.504 |

Strong: capability_breakthrough (F1 0.83), multi_agent_coordination (recall 1.00, but it absorbs other classes).
Weak: autonomous_action (0/2), agentic_development (recall 0.17: 3 of 6 predicted multi_agent_coordination, 2 research_milestone), infrastructure_ecosystem.
The 44-row holdout has classes with 2-3 examples, so one document moves a class F1 by 0.3-0.5. Quote the CV figure beside the holdout one.

## has_event, 5-fold CV over 395 docs
yes / no_not_significant / no_off_topic: accuracy 0.808, macro-F1 0.708 (majority 0.549 / 0.236). yes vs no binary: accuracy 0.820, macro-F1 0.814.
no_off_topic is the weak class (recall 0.32 on 38 docs): off-topic papers share vocabulary ("multi-agent", "planning") with on-topic ones.

## What this tells us
1. Words alone separate category only moderately; the precedence rules (incident > multi-agent > autonomous > adoption > infra > agentic > capability > research) are logic over the main claim, which TF-IDF cannot learn from 173 rows. Next: sentence-transformer embeddings (Colab) and the extraction features (trigger rule, trigger score, system/org presence) as extra inputs.
2. multi_agent_coordination is over-predicted because the word "agent(s)" dominates the corpus; check class weights or add trigger-sentence text as a second field.
3. Numbers are one seed, small data, no tuning of C or n-grams. They are a floor.
