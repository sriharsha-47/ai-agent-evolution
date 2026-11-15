# Demo script (about 8 minutes)

**Before the demo.** Stop any running Streamlit. Build the database and start the app:

```
py -3.13 scripts/build_db.py --events data/annotations/gold_events_v1.csv
py -3.13 scripts/load_extracted.py data/processed/extracted_events.csv
py -3.13 -m streamlit run app/streamlit_app.py
```

Check the Home page shows 3,018 documents and the expected event count before presenting.

## 1. Home (30 seconds)
Say: the corpus has 3,018 dated documents from 2017 to September 2026: arXiv papers, lab blog posts and GitHub
release notes. Point to the documents-per-year table and say agent activity is concentrated after 2022.

## 2. Corpus search (1 minute)
- Open **1 Corpus**. Search `computer use agent`.
- Add a source type filter (lab_blog) and a year range (2024 to 2026).
- Open one result, click **Show full text**, and point to the source link.
Say: every later number can be traced back to documents like this.

## 3. Timeline (2 minutes)
- Open **2 Timeline**. It starts on gold events only.
- Hover a dot to show organization, capability and trigger.
- Filter by category `multi_agent_coordination`, then by autonomy level. Show the monthly bar chart.
- Add `extracted` to the Origin filter and explain it is provisional model output, not labelled by hand.
- Pick an event in **Open an event** and show its sources.

## 4. Phases (1 minute)
- Open **3 Phases**. Show events per month with the 3-month average, then multi-agent share and mean autonomy.
- Show the dominant category per month and say months with fewer than 3 events get no label.
- If `phases.csv` is loaded, point to the change-point lines; otherwise say the phase boundaries come from the
  change-point analysis and are validated against manually reviewed transition periods.

## 5. Forecast and evidence (2 minutes)
- Open **4 Forecast**. Read the predicted category for the next month, the class probabilities and the model name.
- Show the baselines and the test metrics next to the model. State plainly how the model compares with the
  most-frequent baseline. (Fill in the final numbers once the team has chosen the model.)
- Click **Evidence** on a supporting event. The **5 Evidence** page shows the event fields, the source documents and the
  trigger in context.
Say: the forecast is only as good as the labelled months behind it. We report its limits.

## 6. Limits (30 seconds)
News and reports are an imperfect proxy for actual AI development. Few months have enough events, one category
dominates, extraction and linking can contain errors, and the extracted events are provisional. The output is a
data-driven analysis of reported developments, not a guaranteed prediction.

## If something fails
- "Database not found": run `build_db.py` as above.
- Forecast page shows a yellow banner: `data/processed/forecast_latest.json` is missing; the page is showing a
  placeholder baseline.
- Phases page says a file is missing: copy the tables into `results/tables/`.
- `PermissionError` while rebuilding: stop Streamlit first.
