# Reviewed false merges (link_false_merges.csv, Harshita): what they say about extraction

56 reviewed pairs, 18 groups, 53 documents, all labelled N (not the same event). 52 pairs are OpenAI, 3 Anthropic. 32 pairs are on the same day, the rest within 2-7 days.

## Cause on the extraction side
The `model_agent` in the file equals my system-name output for all 53 documents, and the linker merged on (organisation, model_agent, date window). Three problems in my part:
1. **The system name is a model/product name, not an event key.** Launch posts, customer stories, feature posts and system cards about the same model share it (group 975: 15 pairs, all "GPT-4o"; group 437: "GPT-4"). The name is right as "which model", but it cannot decide "same event".
2. **o1 posts were named "ChatGPT"** (the o1 rule missed "OpenAI o1" in titles and the non-breaking hyphen in "o1-preview"). 14 pairs have the generic ChatGPT on one side; the six pairs of group 963 (genetics, economics, coding, o1-mini) merged on it.
3. **Mixed hyphens**: U+2011 in 33 of 56 pairs ("GPT-4" vs "GPT-4" with a different dash), so the same model has several spellings, which makes any exact-match key unreliable in both directions.

## Fixed in `ner.py`
`canon()` normalises Unicode hyphens and non-breaking spaces in the system and organisation helpers; the o1 family (o1, o1-mini, o1-preview, "OpenAI o1" in titles) is recognised. NER evaluation is unchanged (model_agent F1 0.73). With the current names 19 of the 56 false pairs no longer share a system (37 still do).

## What extraction cannot fix (for Saketh and Harshita)
- Of the 56 pairs, 37 still share organisation, system and a date window with the corrected names: group 975 (15), 437 (9) and others are different posts about the same model. The merge rule needs a text signal besides the model name.
- Title TF-IDF cosine of these false pairs has median 0.02 (90th percentile 0.41). A gate at 0.3 would still merge 9 of the 56, at 0.5 still 3; with the name fix plus a 0.3 gate, 5 remain. So a title or embedding gate removes most of them, but this file only contains false merges, so it cannot choose the threshold: a sample of true duplicates (Y pairs) is needed to see what the gate loses.
- Some pairs may be debatable under the guidelines (for example "GPT-5 System Card" and "Introducing GPT-5" are the same release, reviewed N). A rule is needed for system cards and addenda: same event as the launch, or their own event?
- Suggested: the linker uses title similarity as a hard requirement, and customer-story posts are not merged with launch posts. A customer-story detector is not built here (titles like "X uses ChatGPT ..." are too varied for a simple rule; it would be a small classifier with labelled examples).

Reproduce: `python scripts/diagnose_false_merges.py data/interim/corpus_v1_clean.jsonl link_false_merges.csv`.
