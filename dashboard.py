"""
Sentinel dashboard: drift detection results from the saved scenario runs.

Run from the project root (venv active):   streamlit run dashboard.py
"""

import json
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from analyze import ALPHA, METRICS, ORDER, add_distances, load, scan  # noqa: E402

st.set_page_config(page_title="Sentinel", page_icon="🛰️", layout="wide")

ACTIVE = [m for m in METRICS if m != "latency_ms"]   # latency tracks provider load, not output drift
THRESH = ALPHA / len(ACTIVE)                         # Bonferroni across the alert metrics
INJECTED = {
    "control": "Nothing (same settings as baseline) - any alert here is a false alarm",
    "terse": "System prompt: answer in one short sentence",
    "cautious": "System prompt: formal, cautious tone, state uncertainty",
    "model_swap": "Model: gpt-oss-20b -> gpt-oss-120b, same prompt",
    "topic_shift": "Inputs: only health/food keyword questions (a weak shift)",
    "domain_shift": "Inputs: programming questions instead of trivia",
}


@st.cache_data(show_spinner="Computing windows...")
def compute(window: int, step: int):
    base = load("baseline")
    scen = {n: load(n) for n in ORDER}
    scen = {n: r for n, r in scen.items() if r}
    add_distances(base, scen.values())
    cols = ["qid"] + METRICS
    base_df = pd.DataFrame([{c: r.get(c) for c in cols} for r in base])
    out = {}
    for n, rows in scen.items():
        df = pd.DataFrame([{c: r.get(c) for c in cols} for r in rows])
        wins = scan(base, rows, window, step, ACTIVE)
        out[n] = {"df": df, "wins": [{"end": w["end"], "p": w["p"]} for w in wins]}
    return base_df, out


def read_json(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


st.title("🛰️ Sentinel")
st.write("Monitors an LLM's output statistics against a 100-query baseline and flags drift. "
         "Each scenario injects one known change, so we can check whether it gets caught.")

if not os.path.exists("data/runs/baseline.jsonl"):
    st.error("No data/runs/baseline.jsonl found. Run src/collect.py first.")
    st.stop()

with st.sidebar:
    st.header("Detection settings")
    window = st.slider("Window (queries)", 20, 50, 30, 5)
    step = st.slider("Step (queries)", 1, 10, 5)
    confirm = st.slider("Consecutive breaches to alert", 1, 4, 2)
    st.caption(f"Alert metrics: {', '.join(ACTIVE)}. Latency is logged but not used for alerts.")
    st.caption(f"Per-metric threshold: p < {THRESH:.3f} (0.05 / {len(ACTIVE)}, Bonferroni).")

base_df, res = compute(window, step)

# ---------- summary ----------
st.subheader("Did it catch the injected drift?")
rows = []
for n, r in res.items():
    wins = r["wins"]
    if not wins:
        rows.append({"scenario": n, "injected change": INJECTED.get(n, ""), "queries": len(r["df"]),
                     "windows in breach": "too few rows", "alert after (queries)": "-", "metrics": "-"})
        continue
    flagged = [[m for m, p in w["p"].items() if p < THRESH] for w in wins]
    streak, alert_at = 0, None
    for w, f in zip(wins, flagged):
        streak = streak + 1 if f else 0
        if streak >= confirm and alert_at is None:
            alert_at = w["end"]
    fired = sorted({m for f in flagged for m in f})
    rows.append({"scenario": n, "injected change": INJECTED.get(n, ""), "queries": len(r["df"]),
                 "windows in breach": f"{sum(1 for f in flagged if f)}/{len(wins)}",
                 "alert after (queries)": str(alert_at) if alert_at else "no alert",
                 "metrics": ", ".join(fired) or "-"})
st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
st.caption("Windows overlap, so the breach counts are not independent samples. One run per scenario.")

# ---------- drill-down ----------
st.subheader("Drill-down")
names = [n for n in res if res[n]["wins"]]
c1, c2 = st.columns([1, 1])
scenario = c1.selectbox("Scenario", names, index=names.index("terse") if "terse" in names else 0)
metric = c2.selectbox("Metric for the raw trend", ACTIVE + ["latency_ms"])

left, right = st.columns(2)
wins = res[scenario]["wins"]
ev = pd.DataFrame([{"queries seen": w["end"], "metric": m, "evidence": min(20.0, -np.log10(max(p, 1e-300)))}
                   for w in wins for m, p in w["p"].items()])
line = alt.Chart(ev).mark_line(point=True).encode(
    x="queries seen:Q", y=alt.Y("evidence:Q", title="drift evidence, -log10(p), capped at 20"),
    color="metric:N")
rule = alt.Chart(pd.DataFrame({"y": [-np.log10(THRESH)]})).mark_rule(strokeDash=[4, 4], color="red").encode(y="y:Q")
left.markdown("**Drift evidence per window** (above the red line = breach)")
left.altair_chart(line + rule, use_container_width=True)

df = res[scenario]["df"].reset_index(drop=True)
roll = df[metric].astype(float).rolling(10, min_periods=5).mean()
trend = pd.DataFrame({"query": np.arange(1, len(df) + 1), "rolling mean (10)": roll})
base_mean = float(pd.to_numeric(base_df[metric], errors="coerce").mean())
tline = alt.Chart(trend).mark_line().encode(x="query:Q", y=alt.Y("rolling mean (10):Q", title=metric))
brule = alt.Chart(pd.DataFrame({"y": [base_mean]})).mark_rule(strokeDash=[4, 4], color="gray").encode(y="y:Q")
right.markdown(f"**{metric}**: rolling mean vs baseline mean (grey line)")
right.altair_chart(tline + brule, use_container_width=True)

# ---------- alerts ----------
st.subheader("Alert history")
alerts = read_jsonl("results/alerts.jsonl")
summary = {r["scenario"]: r for r in read_json("results/monitor_summary.json", []) if r.get("alert")}
if alerts:
    a = pd.DataFrame(alerts)[["ts", "scenario", "queries_seen", "metrics"]]
    a["delivery ms"] = a["scenario"].map(lambda s: (summary.get(s, {}).get("alert") or {}).get("alert_latency_ms"))
    a["metrics"] = a["metrics"].map(lambda m: ", ".join(m))
    st.dataframe(a.sort_values("ts", ascending=False), hide_index=True, use_container_width=True)
    st.caption("Delivery ms = time for the alert sink (file, plus Slack if configured) to accept the alert "
               "after the breach was confirmed. It is not the time until a person sees it.")
else:
    st.info("No alerts yet. Run `python3 src/monitor.py`.")

with st.expander("What this does and doesn't show", expanded=False):
    st.markdown(
        "- **Replay, not live traffic.** Scenarios were collected earlier and replayed through the detector.\n"
        "- **Not detected:** the model swap (gpt-oss-20b to 120b) produced no breach on these output statistics.\n"
        "- **Weak test:** the health-topic subset is mostly the same kind of question as the baseline, so "
        "barely moving the embeddings is expected.\n"
        "- **Latency is excluded from alerts.** It flagged the no-drift control in most windows because "
        "API response time depends on provider load.\n"
        "- **Small samples, one run each**, with overlapping windows; no confidence intervals."
    )
