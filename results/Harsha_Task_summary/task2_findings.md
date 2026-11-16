# Task 2 findings: event-trigger rules (SH / extraction lane)

Files: `src/ainevo/extraction/segment.py` (sentence splitter, no spaCy), `triggers.py` (rules), `scripts/eval_triggers.py`,
`tests/test_segment.py`, `tests/test_triggers.py` (9 tests pass), `prompts/trigger_extraction.md` (LLM prompt, untested).

## What it does
`find_triggers(text, source_type)` splits the text into sentences (bullets and ## headings count as boundaries), matches
announcement / research / "we've trained" / deal / scale / feature rules, scores each match, and returns ranked candidates
with exact character offsets. `event_score` = best candidate score. Page chrome ("Loading… Share", "Read paper") is masked;
noun uses ("the launch of", "open-source datasets") and maintenance bullets (fix, bump, chore) are ignored.

## Score against my gold triggers (49 events, SH labels, 90 docs)
Hit = the candidate's sentence contains the gold trigger.

| source | top-1 | top-3 | n |
|---|---|---|---|
| lab_blog | 14 | 18 | 24 |
| paper | 16 | 16 | 19 |
| release_notes | 0 | 2 | 6 |
| all | 30 (61%) | 36 (73%) | 49 |

History on the same data: first version 22 / 29 (top-1 / top-3). Fixes for noun matches, page chrome and an abstract-style
rule gave 23 / 30. Adding lead-sentence position (blogs) and Features-vs-Fixes headings (release notes) gave 30 / 36.
**The last step was tuned on these 90 documents, so 73% is optimistic.** Measure again on Harshita's merged gold set or a
fresh batch before quoting it.

## Findings
- Lab blogs state their news early: all 24 gold triggers sit in the first four sentences (14 in the first). Position is the
  strongest single signal.
- Papers work well because "we present / propose / introduce" is a reliable abstract pattern. Misses: abstracts that say
  "Our approach, X, addresses..." or "We observed evidence...".
- Release notes are weak (2 of 6): the gold trigger is one specific feature bullet among many; rules cannot tell which.
  Only 6 examples, so the number is very uncertain.
- Many gold triggers use ordinary verbs ("combine", "brings", "creates", "is trained", "worked together to add"). No
  announcement-verb list covers these; this is where the LLM prompt (or a learned span model) should help.
- The trigger score is NOT a good has_event classifier: AUC 0.71 (yes vs no), 0.77 vs not_significant, 0.54 vs off_topic.
  Off-topic documents use the same verbs. has_event needs the classifier (T6), not this score.

## Limits
- Rules only; weights are hand-set. The LLM prompt has not been run.
- Gold triggers are one annotator's choices (me, for Harsha's sheet); other annotators may choose different spans.
- Evaluated on 90 docs, mostly OpenAI blogs and a few frameworks; arXiv abstracts from other years may differ.

## Next
Run the same eval on the merged gold set after the agreement meeting; try the LLM prompt on the ~13 missed sentences;
then Task 3 (sentence/event segmentation on top of this splitter).
