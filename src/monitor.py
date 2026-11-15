"""
Replay each scenario as a live stream, track it in MLflow, and send an alert on a confirmed breach.

    python3 src/monitor.py                 # all scenarios that have data
    python3 src/monitor.py --scenario terse
    python3 src/monitor.py --confirm 2     # windows in a row that must breach (default 2)

What it does, per scenario:
  * slides a 30-query window over the scenario's rows, one step of 5 queries at a time
  * tests each window against the baseline (same tests as analyze.py, Bonferroni rule,
    latency excluded) and logs every p-value to MLflow (one MLflow run per scenario)
  * when `--confirm` consecutive windows breach, it fires ONE alert and records
    alert_latency_ms = time from the breach decision until the alert was delivered

Alert sinks:
  * always: appended to results/alerts.jsonl
  * if SLACK_WEBHOOK_URL is set in .env: also posted to Slack (this is the delivery time that matters)

Tracking data goes to ./mlflow.db (view it with:  mlflow ui --backend-store-uri sqlite:///mlflow.db)
Note: this replays saved data. 'alert latency' is decision -> delivery, not time-to-detect in real traffic.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import (ALPHA, METRICS, ORDER, add_distances, load, scan)  # noqa: E402

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

DB = "sqlite:///mlflow.db"
ALERTS = "results/alerts.jsonl"


def send_alert(text, record):
    """Deliver an alert. Returns delivery latency in ms (sink round-trip) and the channels used."""
    channels = ["file"]
    t0 = time.perf_counter()
    os.makedirs(os.path.dirname(ALERTS), exist_ok=True)
    with open(ALERTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if url:
        req = urllib.request.Request(url, data=json.dumps({"text": text}).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            r.read()
        channels.append("slack")
    return round((time.perf_counter() - t0) * 1000, 1), channels


def run_scenario(mlflow, name, base, rows, args, active):
    thresh = ALPHA / len(active)
    wins = scan(base, rows, args.window, args.step, active)
    if not wins:
        print(f"{name:<12} only {len(rows)} rows, need >= {args.window}; skipped")
        return None

    result = {"scenario": name, "rows": len(rows), "windows": len(wins), "alert": None}
    streak = 0
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"scenario": name, "window": args.window, "step": args.step,
                           "alpha_bonferroni": round(thresh, 5), "confirm": args.confirm,
                           "alert_metrics": ",".join(active), "n_rows": len(rows),
                           "baseline_rows": len(base)})
        for w in wins:
            fired = [m for m, p in w["p"].items() if p < thresh]
            mlflow.log_metrics({f"p_{m}": p for m, p in w["p"].items()}
                               | {"breach": 1.0 if fired else 0.0}, step=w["end"])
            streak = streak + 1 if fired else 0
            if streak >= args.confirm and result["alert"] is None:
                t_decision = time.perf_counter()
                text = (f":rotating_light: Sentinel: drift confirmed in '{name}' after {w['end']} queries "
                        f"({args.confirm} consecutive windows). Metrics: {', '.join(fired)}")
                record = {"ts": datetime.now(timezone.utc).isoformat(), "scenario": name,
                          "queries_seen": w["end"], "metrics": fired, "text": text}
                try:
                    latency_ms, channels = send_alert(text, record)
                except Exception as e:
                    latency_ms, channels = None, [f"FAILED: {type(e).__name__}"]
                total_ms = round((time.perf_counter() - t_decision) * 1000, 1)
                result["alert"] = {"queries_seen": w["end"], "metrics": fired,
                                   "alert_latency_ms": latency_ms, "channels": channels}
                mlflow.log_metrics({"alert_queries_seen": float(w["end"]),
                                    "alert_latency_ms": float(latency_ms if latency_ms is not None else -1)})
                mlflow.set_tag("alert_channels", ",".join(channels))
                mlflow.set_tag("alert_fired", "true")
        if result["alert"] is None:
            mlflow.set_tag("alert_fired", "false")
        mlflow.log_metric("windows_in_breach", float(sum(
            1 for w in wins if any(p < thresh for p in w["p"].values()))))
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", help="one scenario (default: all that have data)")
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--confirm", type=int, default=2, help="consecutive breaching windows needed to alert")
    ap.add_argument("--exclude", default="latency_ms")
    args = ap.parse_args()

    import mlflow
    mlflow.set_tracking_uri(DB)
    mlflow.set_experiment("sentinel")

    base = load("baseline")
    if len(base) < 50:
        sys.exit("Need data/runs/baseline.jsonl first.")
    names = [args.scenario] if args.scenario else ORDER
    scen = {n: load(n) for n in names}
    scen = {n: r for n, r in scen.items() if r}
    add_distances(base, scen.values())
    excluded = {m for m in args.exclude.split(",") if m}
    active = [m for m in METRICS if m not in excluded]

    print(f"Slack: {'on' if os.environ.get('SLACK_WEBHOOK_URL') else 'off (alerts go to results/alerts.jsonl only)'}")
    print(f"Window {args.window}, step {args.step}, alert after {args.confirm} consecutive breaching windows\n")
    print(f"{'scenario':<12} {'alert at (queries)':>19} {'delivery ms':>12}  metrics")
    out = []
    for name, rows in scen.items():
        r = run_scenario(mlflow, name, base, rows, args, active)
        if not r:
            continue
        out.append(r)
        a = r["alert"]
        if a:
            lat = "n/a" if a["alert_latency_ms"] is None else f"{a['alert_latency_ms']:.0f}"
            print(f"{name:<12} {a['queries_seen']:>19} {lat:>12}  {', '.join(a['metrics'])}  [{'+'.join(a['channels'])}]")
        else:
            print(f"{name:<12} {'no alert':>19}")
    json.dump(out, open("results/monitor_summary.json", "w"), indent=2)
    print("\nView runs:  mlflow ui --backend-store-uri sqlite:///mlflow.db   (then open http://127.0.0.1:5000)")
    print("Wrote results/monitor_summary.json")


if __name__ == "__main__":
    main()
