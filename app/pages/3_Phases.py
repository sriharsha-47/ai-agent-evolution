"""Trends and phases (owner: VV). Reads Saketh's tables from results/tables/.

features_monthly_gold.csv   monthly features, 2022 onwards
counts_quarterly_gold.csv   quarterly counts by category, 2017-2021
changepoints_gold.csv       change-point segments on event counts (segment, start, end, months, mean_*)
changepoints_gold_shares.csv  same, found on category shares
phases.csv (optional)       phase_id, start_month, end_month, label, change_point_month, method (overrides the above)
"""
import pandas as pd
import plotly.express as px
import streamlit as st

from components.data import ROOT

st.set_page_config(page_title="Phases", layout="wide")
st.title("Trends and phases")

TABLES = ROOT / "results" / "tables"
MONTHLY, QUARTERLY, PHASES = (TABLES / "features_monthly_gold.csv", TABLES / "counts_quarterly_gold.csv",
                              TABLES / "phases.csv")

if not MONTHLY.exists():
    st.warning(f"Missing {MONTHLY.relative_to(ROOT)}. Copy Saketh's features_monthly_gold.csv there "
               "(results/tables is not committed by this app).")
    st.stop()

m = pd.read_csv(MONTHLY)
m["when"] = pd.to_datetime(m["month"] + "-01")
st.caption(f"{int(m['n_events'].sum())} gold events across {len(m)} months "
           f"({m['month'].min()} to {m['month'].max()}). Few events per month, so read trends with caution.")

SEGMENT_FILES = {"Event counts": TABLES / "changepoints_gold.csv",
                 "Category shares": TABLES / "changepoints_gold_shares.csv"}


def segments_to_phases(seg: pd.DataFrame, method: str) -> pd.DataFrame:
    """Turn change-point segments into the phase table the charts use."""
    out = pd.DataFrame({"phase_id": seg["segment"], "start_month": seg["start"], "end_month": seg["end"],
                        "label": "Segment " + seg["segment"].astype(str)})
    out["change_point_month"] = [None] + list(seg["start"].iloc[1:])  # a segment starts where the change is
    out["method"] = method
    return out


segments = None
if PHASES.exists():
    phases = pd.read_csv(PHASES)
else:
    available = {k: v for k, v in SEGMENT_FILES.items() if v.exists()}
    phases = None
    if available:
        with st.sidebar:
            choice = st.radio("Change-point method", list(available), help="Both are Saketh's change-point "
                              "analyses on the gold events.")
        segments = pd.read_csv(available[choice])
        phases = segments_to_phases(segments, f"change-point on {choice.lower()}")


def add_phase_marks(fig):
    """Phase bands and change points, only when Saketh's phases.csv exists."""
    if phases is None:
        return fig
    for _, p in phases.iterrows():
        fig.add_vrect(x0=pd.to_datetime(p["start_month"] + "-01"), x1=pd.to_datetime(p["end_month"] + "-01"),
                      opacity=0.08, line_width=0, annotation_text=str(p.get("label", p["phase_id"])),
                      annotation_position="top left")
        cp = p.get("change_point_month")
        if isinstance(cp, str) and cp:
            fig.add_vline(x=pd.to_datetime(cp + "-01"), line_dash="dash")
    return fig


if phases is None:
    st.info("No phase boundaries yet (no phases.csv or changepoints_gold.csv in results/tables). "
            "The charts show raw features only.")
elif segments is not None:
    st.subheader("Change-point segments")
    st.dataframe(segments, hide_index=True, use_container_width=True)
    st.caption("A change point is a sustained shift in the series, found by statistics on a small number of "
               "months. Treat the boundaries as approximate, and as a candidate to check against the reviewed "
               "transition periods.")

st.subheader("Events per month")
m["moving_avg_3m"] = m["n_events"].rolling(3, min_periods=1).mean()
fig = px.bar(m, x="when", y="n_events", height=320, labels={"when": "", "n_events": "events"})
fig.add_scatter(x=m["when"], y=m["moving_avg_3m"], mode="lines", name="3-month average")
st.plotly_chart(add_phase_marks(fig), use_container_width=True)

st.subheader("Feature trends")
options = {"multi_agent_share": "Multi-agent share", "incident_share": "Incident share",
           "mean_autonomy": "Mean autonomy level", "high_autonomy_share": "High-autonomy share",
           "entropy": "Category diversity (entropy)", "novelty": "Novelty", "n_orgs": "Distinct organizations"}
chosen = st.multiselect("Features", [k for k in options if k in m.columns],
                        default=[k for k in ("multi_agent_share", "mean_autonomy") if k in m.columns],
                        format_func=options.get)
if chosen:
    fig = px.line(m, x="when", y=chosen, height=340, labels={"when": "", "value": "value"})
    st.plotly_chart(add_phase_marks(fig), use_container_width=True)

st.subheader("Dominant category per month")
labelled = m.dropna(subset=["dominant_category"])
st.caption(f"{len(labelled)} of {len(m)} months have a label (months with fewer than 3 events are not labelled).")
st.dataframe(labelled[["month", "n_events", "dominant_category"]], hide_index=True, use_container_width=True)

if QUARTERLY.exists():
    st.subheader("Long timeline, 2017-2021 (quarterly)")
    q = pd.read_csv(QUARTERLY)
    cats = [c for c in q.columns if c not in ("quarter", "n_events")]
    long = q.melt(id_vars="quarter", value_vars=cats, var_name="category", value_name="events")
    st.plotly_chart(px.bar(long, x="quarter", y="events", color="category", height=320),
                    use_container_width=True)
else:
    st.caption("counts_quarterly_gold.csv not found, so the 2017-2021 view is hidden.")
