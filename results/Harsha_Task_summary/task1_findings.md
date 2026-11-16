# Task 1 findings: NER + date extraction (SH / extraction lane)

Tools tested on the 20-document sample (`data/interim/sample_20.jsonl`, arXiv abstracts, all published 2026-10-19):
- spaCy 3.8 `en_core_web_sm` + a regex gazetteer for AI labs and model/framework names (`ner.py`)
- Hugging Face `dslim/bert-base-NER` (`scripts/hf_ner_test.py`, run on Google Colab; Windows Application Control blocks spaCy on the laptop)
- Dates: regex + spaCy DATE + dateparser (`temporal.py`)

## 1. NER comparison on the 20 abstracts

| | spaCy + rules | Hugging Face (bert-base-NER) |
|---|---|---|
| Rows | 77 (19 rules, 58 spaCy) | 142 |
| Whole-word entities | 77 | 48 (94 rows, 66%, are word fragments: `##ness`, `GP`, `Tande`) |
| Useful entities | rules ~19/19; spaCy ~2/58 (USGS, Snapdragon) | ~22 of 48 whole-word (~45%) |
| Model names with version | Correct (GPT-6, GPT-5.2, Claude Opus 4.6, Qwen3.5) | Version dropped (`GPT`), `Claude Opus` labelled PER |
| Paper system names | Mostly missed or tagged ORG/PERSON/GPE | Found: WeAct, AllPath, Tandem, BetaMem, RefCraft, ProteinGeoBench, "Agent World Model in the Wild" |
| Products / orgs | USGS, Snapdragon | USGS, Snapdragon 8 Elite, Eclipse Dataspace, LangGraph, Codex, EDC |

Reading of the results:
- The fragments came from my script (`aggregation_strategy="simple"` splits unfamiliar names into word pieces). The script now uses `"max"` (word-level) and slices the surface text from the original string. The committed CSV is the *old* output; a re-run on Colab should remove most fragments but has not been done.
- Hugging Face is the better finder of **paper system names**, which become `model_agent`. Its labels are unstable (the same name appears as MISC and ORG), so use it label-agnostic.
- Rules are the better finder of **model names with versions**, which Hugging Face truncates.
- spaCy `sm` on abstracts is mostly noise (method names such as SFT/LoRA/Sokoban tagged ORG). Its only added value is a few real orgs/products, which Hugging Face also finds.
- Remaining Hugging Face noise: generic words (Standard, Adapter, Nash, Markov, Machine Learning, Val, Bench).
- No tool can supply `organization` for papers: abstracts do not name the lab.

## 2. Dates (20 abstracts and 90 labelled docs)
- 20 abstracts: 1 date found (a bare year). Abstracts do not state event dates, so this sample cannot test date extraction. Fall back to publication_date.
- 90 labelled docs, date found in text: lab_blog 33/37, release_notes 7/29, paper 1/24.
- The 49/49 match with my `event_date` labels is circular (blog dates were copied from the same header line). It is not evidence.
- Risky "first absolute date" cases: an AutoGPT note says "Date: April 2024" but was published 2025-04-15 (source typo); an AutoGen note mentions 2024-12-17 but was published 2025-07-01 (an earlier event).
- `dateparser` crashes under the `Asia/Calcutta` timezone unless `TIMEZONE="UTC"` is set (already fixed in `temporal.py`).

## 3. Organisation / model_agent recall on the 90 labelled docs (spaCy + rules)
- organization 26/30 (lenient substring). Misses are lab blogs that never write "OpenAI" in the text.
- model_agent: real misses are paper system names (ATLAS, QAgent, ClawVM, REALM-Bench, AMA), which Hugging Face and the title heuristic below can cover.
- Release notes: spaCy returns commit-message junk ("chore(core", "harden", "CI").

## 4. Decisions / recommended design for Task 2 onward
1. Model/framework names: keep the rule gazetteer (add names as they appear).
2. Paper system names: use the title text before the first ':' plus Hugging Face (word-level aggregation) as candidates; drop generic words with a stoplist.
3. Organisation for blogs and release notes: take it from the `source` field (OpenAI; repo owner), not NER.
4. Do not run statistical NER on release notes; rules on title + text only.
5. Event date: blogs use the header date; release notes and papers use publication_date.
6. Drop spaCy `sm` from the paper path unless a re-run shows it adds anything beyond Hugging Face.
