"""
Drift detection: compare sliding windows of each scenario against the baseline.

    python3 src/analyze.py                      # all scenarios that have data
    python3 src/analyze.py --window 30 --step 5

Per window and per metric: Kolmogorov-Smirnov test vs the 100 baseline rows
(Fisher exact test for the 0/1 refusal flag). Two alert rules are reported:
  A) any metric with p < 0.05           (the plain rule)
  B) any metric with p < 0.05 / 6       (Bonferroni: stricter, fewer false alarms)

Detection delay = number of queries seen when the first window is flagged.
The 'control' scenario has the SAME settings as the baseline, so any alert on it is a
false alarm; that is how the false-alarm rate is measured.

Writes results/detection.json.
"""

import argparse
import json
import os

import numpy as np
from scipy.stats import fisher_exact, ks_2samp

METRICS = ["response_words", "completion_tokens", "latency_ms",
           "hedge_rate", "refusal", "embedding_distance"]
ORDER = ["control", "terse", "cautious", "model_swap", "topic_shift", "domain_shift"]
ALPHA = 0.05


def load(name, d="data/runs"):
    path = os.path.join(d, f"{name}.jsonl")
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    return sorted(rows, key=lambda r: r["qid"])


def _dist(e, c):
    c = c / (np.linalg.norm(c) or 1.0)
    return float(1.0 - np.dot(e, c) / (np.linalg.norm(e) or 1.0))


def add_distances(base_rows, scenario_rows):
    """Topic-embedding drift: cosine distance to the baseline centroid.
    Baseline rows use a leave-one-out centroid so they are not compared with themselves."""
    B = np.array([r["embedding"] for r in base_rows])
    total, n = B.sum(axis=0), len(B)
    for r, e in zip(base_rows, B):
        r["embedding_distance"] = _dist(e, (total - e) / (n - 1))
    for rows in scenario_rows:
        for r in rows:
            r["embedding_distance"] = _dist(np.array(r["embedding"]), total / n)


def pvalue(metric, base, win):
    base = np.array([x for x in base if x is not None], dtype=float)
    win = np.array([x for x in win if x is not None], dtype=float)
    if len(base) < 5 or len(win) < 5:
        return 1.0
    if metric == "refusal":
        table = [[int(win.sum()), len(win) - int(win.sum())],
                 [int(base.sum()), len(base) - int(base.sum())]]
        return float(fisher_exact(table)[1])
    return float(ks_2samp(base, win).pvalue)


def scan(base_rows, rows, window, step, active):
    base = {m: [r.get(m) for r in base_rows] for m in active}
    out = []
    for end in range(window, len(rows) + 1, step):
        w = rows[end - window: end]
        p = {m: pvalue(m, base[m], [r.get(m) for r in w]) for m in active}
        out.append({"end": end, "p": p})
    return out


def summarize(wins, n_metrics):
    def rule(thresh):
        flagged = [w for w in wins if any(v < thresh for v in w["p"].values())]
        first = flagged[0]["end"] if flagged else None
        counts = {}
        for w in flagged:
            for m, v in w["p"].items():
                if v < thresh:
                    counts[m] = counts.get(m, 0) + 1
        return {"windows_flagged": len(flagged), "windows": len(wins), "first_alert_at": first,
                "metrics": dict(sorted(counts.items(), key=lambda kv: -kv[1]))}
    return {"A_p<0.05": rule(ALPHA), "B_bonferroni": rule(ALPHA / n_metrics)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--window", type=int, default=30)
    ap.add_argument("--step", type=int, default=5)
    ap.add_argument("--exclude", default="latency_ms",
                    help="comma-separated metrics NOT used for alerts (default: latency_ms, because "
                         "API latency tracks provider load / time of day, not model behaviour). Use '' to include all.")
    args = ap.parse_args()
    excluded = {m for m in args.exclude.split(",") if m}
    active = [m for m in METRICS if m not in excluded]

    base = load("baseline")
    if len(base) < 50:
        raise SystemExit("Need data/runs/baseline.jsonl (100 rows). Run collect.py first.")
    scen_rows = {n: load(n) for n in ORDER}
    scen_rows = {n: r for n, r in scen_rows.items() if r}
    add_distances(base, scen_rows.values())
    results = {}

    # baseline self-check: if the first half already differs from the second half,
    # the baseline is not stable and alerts on that metric are less trustworthy
    h = len(base) // 2
    print(f"Baseline: {len(base)} rows. Self-check (first {h} vs last {len(base) - h}), p-values:")
    sc = {m: pvalue(m, [r.get(m) for r in base[:h]], [r.get(m) for r in base[h:]]) for m in METRICS}
    print("  " + "  ".join(f"{m}={v:.3f}" + ("*" if v < ALPHA else "") for m, v in sc.items()))
    print("  (* = the baseline itself shifts on this metric)\n")
    results["baseline_self_check"] = sc

    print(f"Alert metrics: {', '.join(active)}" + (f"   (not used for alerts: {', '.join(sorted(excluded))})" if excluded else ""))
    print(f"Window={args.window} queries, step={args.step}. 'first alert' = queries seen when first flagged.\n")
    print(f"{'scenario':<12} {'rule':<14} {'flagged':>9} {'first alert':>12}   metrics that fired")
    for name, rows in scen_rows.items():
        wins = scan(base, rows, args.window, args.step, active)
        if not wins:
            print(f"{name:<12} (only {len(rows)} rows, need >= {args.window})")
            continue
        summ = summarize(wins, len(active))
        results[name] = {"n_rows": len(rows), **summ}
        for rule, s in summ.items():
            fa = "none" if s["first_alert_at"] is None else str(s["first_alert_at"])
            top = ", ".join(f"{m}({c})" for m, c in s["metrics"].items()) or "-"
            print(f"{name:<12} {rule:<14} {s['windows_flagged']:>4}/{s['windows']:<4} {fa:>12}   {top}")
        print()
    print("control = same settings as baseline, so every flagged window there is a false alarm.")

    os.makedirs("results", exist_ok=True)
    json.dump(results, open("results/detection.json", "w"), indent=2)
    print("Wrote results/detection.json")


if __name__ == "__main__":
    main()
