"""
Send questions through the monitored LLM and log one JSON line per call.

    python3 src/collect.py --scenario baseline          # first 100 questions, normal settings
    python3 src/collect.py --scenario control           # next 100, SAME settings (no drift expected)
    python3 src/collect.py --scenario baseline --n 5    # quick test

Output: data/runs/<scenario>.jsonl  (one line per question; safe to stop and rerun, it resumes)

Questions come from data/truthfulqa_subset.json, shuffled with a fixed seed so every
scenario draws a different, reproducible slice.
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import text_metrics  # noqa: E402

MODEL = "openai/gpt-oss-20b"
BASE_SYSTEM = "Answer the question concisely in 2-3 sentences."

# name -> settings. start/n pick a slice of the shuffled question list.
# baseline + control use identical settings (control = "no drift" reference).
# The others inject one known change each, so we can test whether it gets detected.
SCENARIOS = {
    "baseline": {"start": 0,   "n": 100, "model": MODEL, "system": BASE_SYSTEM},
    "control":  {"start": 100, "n": 100, "model": MODEL, "system": BASE_SYSTEM},
    # style drift: much shorter answers
    "terse":    {"start": 200, "n": 100, "model": MODEL, "system": "Answer in one short sentence."},
    # tone drift: more hedged, cautious wording
    "cautious": {"start": 300, "n": 100, "model": MODEL,
                 "system": "Answer in a formal, cautious tone. State any uncertainty explicitly."},
    # model drift: same prompt, different model (a subtle one)
    "model_swap": {"start": 400, "n": 100, "model": "openai/gpt-oss-120b", "system": BASE_SYSTEM},
    # input drift (clear): programming questions, a different domain from TruthfulQA
    "domain_shift": {"start": 0, "n": 68, "model": MODEL, "system": BASE_SYSTEM,
                     "questions_file": "data/code_questions.json"},
    # input drift (weak): only health/food/body questions from the unused part of the list
    "topic_shift": {"start": 500, "n": 100, "model": MODEL, "system": BASE_SYSTEM,
                    "filter": r"\b(health|eat|food|drink|doctor|cancer|vaccine|disease|sleep|diet|brain|body|blood|pain|medic)"},
}


def load_questions(path="data/truthfulqa_subset.json", seed=0):
    rows = json.load(open(path))
    random.Random(seed).shuffle(rows)
    return [r["question"] for r in rows]


def pick(questions, cfg):
    """(qid, question) pairs for a scenario. qid = index in the shuffled list."""
    if cfg.get("questions_file"):
        qs = json.load(open(cfg["questions_file"]))
        return [(10000 + i, q) for i, q in enumerate(qs)][: cfg["n"]]
    if cfg.get("filter"):
        rx = re.compile(cfg["filter"], re.I)
        pool = [(i, q) for i, q in enumerate(questions) if i >= cfg["start"] and rx.search(q)]
        return pool[: cfg["n"]]
    return list(enumerate(questions[cfg["start"]: cfg["start"] + cfg["n"]], start=cfg["start"]))


def done_ids(path):
    ids = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line:
                ids.add(json.loads(line)["qid"])
    return ids


def call_llm(client, model, system, question):
    """One monitored call. Returns (answer, latency_ms, completion_tokens)."""
    t0 = time.perf_counter()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": question}],
        max_completion_tokens=1500,   # reasoning models spend tokens on thinking first
        reasoning_effort="low",
        temperature=0.7,
    )
    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    answer = (resp.choices[0].message.content or "").strip()
    tokens = getattr(getattr(resp, "usage", None), "completion_tokens", None)
    return answer, latency_ms, tokens


def run_scenario(client, embed, name, cfg, questions, out_dir="data/runs", n=None, log=print):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{name}.jsonl")
    todo = pick(questions, cfg)[: (n or cfg["n"])]
    log(f"{name}: {len(todo)} questions selected")
    seen = done_ids(path)
    written = 0
    for qid, q in todo:
        if qid in seen:
            continue
        try:
            answer, latency_ms, tokens = call_llm(client, cfg["model"], cfg["system"], q)
        except Exception as e:
            msg = str(e).lower()
            if "429" in msg or "rate_limit" in msg or "rate limit" in msg:
                log("Rate/daily limit reached. Progress is saved; rerun this command later.")
                return written, True
            log(f"[{qid}] error: {type(e).__name__}: {str(e)[:120]}")
            continue
        if not answer:
            log(f"[{qid}] empty answer, skipped (rerun to retry)")
            continue
        row = {
            "qid": qid, "scenario": name, "model": cfg["model"],
            "ts": datetime.now(timezone.utc).isoformat(),
            "latency_ms": latency_ms, "completion_tokens": tokens,
            **text_metrics(answer),
            "embedding": [round(float(x), 4) for x in embed(answer)],
        }
        with open(path, "a", encoding="utf-8") as f:   # append per row: nothing is lost on stop
            f.write(json.dumps(row) + "\n")
        written += 1
        log(f"[{name} {qid}] {latency_ms:>7.0f} ms  {row['response_words']:>3} words  refusal={row['refusal']}")
    return written, False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenario", required=True, choices=sorted(SCENARIOS))
    ap.add_argument("--n", type=int, help="only run this many (for a quick test)")
    args = ap.parse_args()

    from dotenv import load_dotenv
    load_dotenv()
    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not found. Put it in .env in this folder.")
    from groq import Groq
    from sentence_transformers import SentenceTransformer

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    st = SentenceTransformer("all-MiniLM-L6-v2")
    embed = lambda text: st.encode(text, normalize_embeddings=True)

    questions = load_questions()
    written, hit_limit = run_scenario(client, embed, args.scenario, SCENARIOS[args.scenario], questions, n=args.n)
    total = len(done_ids(os.path.join("data/runs", f"{args.scenario}.jsonl")))
    target = len(pick(questions, SCENARIOS[args.scenario])[: (args.n or SCENARIOS[args.scenario]["n"])])
    print(f"\nWrote {written} new rows. {args.scenario}: {total}/{target} done.")


if __name__ == "__main__":
    main()
