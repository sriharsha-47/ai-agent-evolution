# Task 3 findings: sentence / event segmentation prototype (SH / extraction lane)

Files: `src/ainevo/extraction/segment.py` (`split_sentences` from Task 2 + new `segment_events`), small edits to `triggers.py`
(candidates now carry `sent_idx` and `section`) and `temporal.py` (dateparser imported lazily so `regex_dates` works where its
compiled dependencies are blocked), `scripts/run_event_segmentation.py`, `scripts/eval_segmentation.py`,
`tests/test_segment.py` (13 tests pass), `results/tables/event_units_sample20.csv` (ids/offsets only, no document text).

## What it does
`segment_events(text, source_type)` returns candidate **event units** in document order:
- unit = a sentence whose best trigger scores >= `min_score` (default 3.0) + up to 2 following sentences, stopping at a heading
  or at the next trigger sentence (a new event);
- release notes: each feature bullet is its own unit (no follow-on context); fix/chore bullets are never units;
- every unit carries trigger text + offsets, rule, score, section heading, regex dates found inside it, `rank` and `is_primary`;
- at most `max_units` (default 5) units are kept.

## Evaluation (90 labelled docs; 49 gold events = one gold trigger per document; SH labels)
Threshold curve (max_units = 5):

| min_score | primary unit contains gold trigger | any unit contains it | event docs with a unit | non-event docs with a unit | units/doc event / non-event |
|---|---|---|---|---|---|
| 0 | 33/49 | 40/49 | 47/49 | 36/41 | 3.1 / 2.9 |
| **3.0** | **33/49 (67%)** | **39/49 (80%)** | 47/49 | 32/41 | 2.9 / 2.3 |
| 3.5 | 32/49 | 38/49 | 46/49 | 28/41 | 2.8 / 2.2 |
| 4.0 | 30/49 | 34/49 | 39/49 | 18/41 | 2.0 / 0.9 |
| 5.0 | 16/49 | 20/49 | 21/49 | 8/41 | 0.7 / 0.3 |

By source at 3.0 (primary / any / mean units): lab_blog 16/24, 20/24, 3.6 units; paper 17/19, 17/19, 1.5 units;
release_notes 0/6, 2/6, 4.3 units.

## Findings
1. **Papers segment cleanly** (primary unit right for 17/19, about 1.5 units each: one contribution per abstract).
2. **Blogs: right unit usually found (20/24 any, 16/24 primary)** but 3.6 units per document on average; extra units are
   secondary announcements, history ("we launched X in 2022") or customer examples. Treat rank 1 as the event.
3. **Release notes remain weak** (primary 0/6, any 2/6) with 4.3 units each: the segmenter splits bullets correctly but cannot
   say which bullet is the headline. Only 6 gold examples; do not read much into the number.
4. **Segmentation cannot decide has_event.** At 3.0, 32 of 41 non-event documents still produce a unit. Raising the threshold
   to 4.0 cuts that to 18/41 but loses 8 of 47 event documents. The decision belongs to the classifier (T6).
5. **The impact evidence is usually outside the unit.** Gold `impact` numbers appear in the primary unit for only 10/24
   documents, but in the full text for 24/24. A longer window helps papers (result sentences end the abstract) but not blogs:
   follow = 2 / 4 / 8 / 15 sentences gives 10 / 12 / 15 / 16 of 24 (paper 3 / 5 / 7 / 7 of 10; blog 7 / 7 / 8 / 9 of 14), with
   units growing from about 500 to 1,070 characters. Recommendation: use the unit for trigger / date / category features, and
   search the whole document (or the whole abstract for papers) for impact numbers.

## Limits
- Scores and defaults were chosen on these 90 documents, so the percentages are optimistic; re-measure on the merged gold set.
- One gold event per document, from one annotator. The gold set has no segment boundaries, so "unit boundaries" are only checked
  indirectly (contains trigger, contains impact numbers, units per document).
- Rule-based; the LLM prompt from Task 2 is still untested.

## Next (plan D4-D5)
Review the pilot with Harshita and adjust trigger rules; baseline NER evaluation on the annotated pilot (compare `organization`,
`model_agent` columns with `ner.py` output; apply the Task 1 fixes: org from `source`, system name from title prefix).
