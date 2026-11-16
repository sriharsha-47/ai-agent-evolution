# Task 5 (D5 + D6): baseline NER evaluation and extraction errors

Gold: SH sheet, rows with `has_event = yes` (7 pilot events, 49 events in total, from the 90 labelled docs).
Fields scored: `organization`, `model_agent`. Lenient match (normalised containment). Set-level micro P/R/F1; `doc_hit` = docs where at least one gold item was found.
Run: `python scripts/eval_ner.py <corpus.jsonl> <annotation_SH.xlsx> [--spacy]`  ->  `results/tables/task5_ner_scores.csv`, `task5_ner_errors.csv`.

## Results (all 49 events)

| field | variant | pred | P | R | F1 | doc_hit |
|---|---|---|---|---|---|---|
| organization | rules (gazetteer) | 45 | 0.56 | 0.71 | 0.62 | 23/30 |
| organization | rules + spaCy (Task 1) | 465 | 0.07 | 0.79 | 0.13 | 24/30 |
| organization | **fixed** | 31 | **0.94** | 0.85 | **0.89** | 28/30 |
| model_agent | rules | 99 | 0.32 | 0.39 | 0.35 | 20/49 |
| model_agent | rules + spaCy (Task 1) | 147 | 0.23 | 0.43 | 0.30 | 22/49 |
| model_agent | **fixed** | 40 | **0.85** | 0.65 | **0.73** | 33/49 |

Pilot only (7 events): organization F1 0.80 (rules and fixed), model_agent 0.13 (rules) -> 0.61 (fixed). Seven rows is too small to trust; read it as a sanity check only.
Papers have no gold organization (abstention is correct): fixed abstains on 19/19, rules+spaCy only 4/19.

## What "fixed" is
- organization = `source` field (OpenAI for blogs, GitHub owner for release notes, none for papers) + gazetteer orgs in the title (partners).
- model_agent = title system name (paper: prefix before ':' if <= 3 words; release notes: repo + tag; blogs: earliest gazetteer model in title) + the single most frequent gazetteer model in title + first 600 chars.
- Code: `owner_from_source`, `title_system_name`, `guess_organizations`, `guess_systems` in `src/ainevo/extraction/ner.py` (no spaCy needed).

## Extraction errors (D6), fixed variant, all 49 events
| reason | count | examples |
|---|---|---|
| org: partner or customer is the actor, not in gazetteer | 5 | P06 Stack Overflow, P12 ServiceNow, SH-044 SchoolAI, SH-060 ENEOS Materials, SH-051 DeepMind (body only) |
| model: gold is a descriptive phrase / expanded name | 10 | SH-005, SH-011, SH-014, SH-015, SH-022, SH-040, SH-041, SH-051, P07, SH-007 |
| model: abstained (name exists but no rule fires) | 6 | P13, SH-006 WorldEvolver, SH-013 RBRs, SH-016 RLTR, SH-041 OpenAI Frontier, SH-005 |
| model: wrong pick | 2 | P06 (OverflowAI vs ChatGPT), SH-075 (InstructGPT vs GPT-3) |
| model: false positives (mentioned model is not the event subject) | 6 | SH-007 DALL-E, SH-015 GPT-3, SH-022 MCP, SH-043 ChatGPT, SH-075 GPT-3 |

Findings that matter for the next steps:
1. spaCy ORG alone is unusable as the organisation extractor on this corpus (465 predictions for 34 gold); the corpus `source` field is the right signal for blogs and release notes.
2. Many `model_agent` gold values are descriptive labels I wrote under the guidelines ("Kubernetes cluster ...", "Weak-to-strong generalization (...)"). No NER can match these. Suggest Harshita/the guidelines separate `system_name` (a proper name) from `capability` and keep descriptions out of `model_agent`; until then F1 for this field has a ceiling around 0.8.
3. Customer-story posts (SH-044, SH-060) have the customer as organization in my gold and OpenAI is the source. Guideline v0 should say whether the lab or the customer is the actor.
4. Paper methods named in the abstract but not in the title (WorldEvolver, RLTR) need a pattern like "we propose X" / "named X"; this is the first thing to add to the rules.
5. Caveat: the fixed rules were written after reading these 90 docs, and `title_system_name` got two small tweaks after the first run (earliest title match, case-insensitive `o1 System`). The numbers are therefore optimistic. A fair number needs fresh labelled docs (the next annotation batch).
