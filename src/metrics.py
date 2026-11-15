"""
Per-response metrics for Sentinel.

Five are computed here for every LLM call. The sixth (topic embedding drift) needs a
baseline, so it is computed later at detection time from the stored embedding.

  response_words      length of the answer in words
  completion_tokens   tokens the model generated (from the API usage field)
  latency_ms          wall-clock time of the API call
  refusal             1 if the answer looks like a refusal / "I don't know", else 0
  hedge_rate          hedging words per 100 words (a rough uncertainty proxy)
  embedding_distance  (later) cosine distance from the baseline centroid
"""

import re

REFUSAL = re.compile(
    r"\b(i can'?t|i cannot|i can not|i'?m sorry|i am sorry|i don'?t know|i do not know|"
    r"i'?m not sure|unable to|not able to)\b", re.I)

HEDGES = {
    "may", "might", "could", "possibly", "perhaps", "likely", "unlikely", "probably",
    "generally", "typically", "usually", "often", "sometimes", "unclear", "uncertain",
    "reportedly", "apparently", "seems", "appears", "suggests", "roughly", "approximately",
}


def text_metrics(answer: str) -> dict:
    words = re.findall(r"[A-Za-z']+", answer or "")
    n = len(words)
    hedges = sum(1 for w in words if w.lower() in HEDGES)
    return {
        "response_words": n,
        "refusal": 1 if REFUSAL.search(answer or "") else 0,
        "hedge_rate": round(100.0 * hedges / n, 3) if n else 0.0,
    }
