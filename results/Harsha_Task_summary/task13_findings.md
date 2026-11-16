# Task 13 (D13): category classifier error analysis

Model: TF-IDF + LR (the embedding model needs the npz; re-run with `--emb` for it). Out-of-fold over all 217 gold events (5-fold), so each document is predicted by a model that never saw it: accuracy 0.590, macro-F1 0.504, top-2 accuracy 0.802 (the right class is in the top two 80% of the time).
Run: `python scripts/error_analysis_classifier.py data/interim/corpus_v1_clean.jsonl <gold_events_v1_1.csv>`. Files: `results/tables/d13_confusion.png|csv`, `d13_errors.csv` (all 89 errors), `d13_report.txt`.

## What the confusion matrix shows
- **multi_agent_coordination is the attractor.** Biggest confusions: agentic_development -> multi_agent 12, research_milestone -> multi_agent 11, autonomous_action -> multi_agent 6. The words "multi-agent / agents" dominate the corpus; precedence order puts multi_agent second, so annotators use it for real multi-agent methods but TF-IDF uses it for any text mentioning agents.
- **autonomous_action is never predicted** (0 of 11): its examples share no vocabulary ("Autonomous ..." appears in only 7 of the 30 titles containing "autonom"; the others are multi_agent 13, agentic_development 6, research 3, incident 1). A title keyword rule would not work.
- **infrastructure_ecosystem is spread out**: it goes to agentic_development 5 and capability_breakthrough 5; adoption_scale goes to infrastructure 5 (stated usage vs platform launch is a judgement the text does not make explicit).
- Errors are balanced against the precedence order (44 predict a higher-precedence class than gold, 45 a lower one), so the model is not simply missing or over-applying the precedence rule.

## Label consistency matters more than the model here
1. **Annotators differ.** Out-of-fold accuracy by annotator: SH 0.69, HA 0.64, TS 0.51, VV 0.42 (n = 42/95/35/38). With these sizes it is suggestive, not proof, but it means a model finds some annotators' labels harder to reproduce, i.e. rules are applied differently. Rows with annotator confidence 1 are the worst (0.45, n = 51) versus 0.64 for confidence 2.
2. **Same project, three labels.** microsoft/autogen releases are labelled agentic_development 2, infrastructure_ecosystem 2, multi_agent_coordination 1; AutoGPT releases agentic 6 / infrastructure 3. The guideline for release notes ("changes what agents can do -> agentic_development/infrastructure_ecosystem") does not decide between them. Release notes top the most-confident-wrong list (SH-019, HA-042, HA-054, SH-023).
3. **Multi-agent benchmarks.** REALM-Bench (SH-025), TAMAS (HA-025) and similar are research_milestone in gold but the guideline also says "an LLM multi-agent method -> multi_agent_coordination". Across papers with "multi-agent" in the title: multi_agent 49, research_milestone 8, agentic_development 1. A decision for docs/decisions.md is needed: does a multi-agent benchmark count as multi-agent coordination or as a research milestone?
4. A candidate rule that holds on the gold: every event with "benchmark" in the title is research_milestone (7 of 7). It is based on seven rows of the same data, so it is a hypothesis for Harshita, not something adopted in code.

## Suggestions
- Ask Harshita to rule on (2) and (3) above in docs/decisions.md and re-check the affected labels (list in `d13_errors.csv`; start with the 12 most confident errors in `d13_report.txt`).
- Do not expect category accuracy much above about 0.65 until those rules are fixed; the extra gain from embeddings (about 0.60-0.64) is within noise.
- Re-run this script with `--emb` after the Colab embeddings exist, and after any label fixes.
