"""
Confidence interval computation.

Level 1 (build time): Wilson score CI on per-edge evidence counts.
Level 2 (query time): LLM-elicited epistemic CI from source passages.
Level 3 (optional):   Log-probability CI if provider exposes logprobs.
"""
from __future__ import annotations

import math
from typing import Any

from src.config import AGENT_MODEL, USE_LOGPROB_CI


def wilson_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a proportion k/n.

    k: number of chunks asserting the relationship
    n: total chunks co-mentioning both endpoint entities
    Returns (lower, upper) CI at confidence level 1-alpha.
    """
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        return (0.0, 0.0)
    if k == n:
        return (1.0, 1.0)

    # z for two-tailed 95% CI
    z = _z_score(1.0 - alpha / 2.0)
    p_hat = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p_hat + z2 / (2 * n)) / denom
    margin = (z / denom) * math.sqrt(p_hat * (1 - p_hat) / n + z2 / (4 * n * n))

    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return (round(lower, 4), round(upper, 4))


def _z_score(p: float) -> float:
    """Inverse CDF of standard normal using scipy if available, else approximation."""
    try:
        from scipy.stats import norm  # type: ignore
        return float(norm.ppf(p))
    except ImportError:
        # Rational approximation (Abramowitz & Stegun 26.2.17)
        t = math.sqrt(-2.0 * math.log(1.0 - p))
        c = (2.515517, 0.802853, 0.010328)
        d = (1.432788, 0.189269, 0.001308)
        return t - (c[0] + c[1] * t + c[2] * t**2) / (1 + d[0] * t + d[1] * t**2 + d[2] * t**3)


def compute_edge_confidences(db: Any) -> int:
    """Compute Wilson CIs for all relationship edges in the graph.

    For each edge, count:
      k = edge.evidence_count
      n = number of chunks mentioning BOTH endpoint entities

    Writes confidence_lower and confidence_upper to each edge.
    Returns number of edges updated.
    """
    from src.graph import RELATIONSHIP_TYPES

    updated = 0
    for rel_type in RELATIONSHIP_TYPES:
        edges = db.run(
            f"MATCH (s)-[r:{rel_type}]->(t) "
            "WHERE r.evidence_count IS NOT NULL "
            "RETURN id(r) AS rid, s.name AS src, t.name AS tgt, "
            "labels(s)[0] AS src_label, labels(t)[0] AS tgt_label, "
            "r.evidence_count AS k, r.source_chunk_ids AS chunk_ids"
        )
        for edge in edges:
            k = edge["k"] or 1
            src_label = edge["src_label"]
            tgt_label = edge["tgt_label"]
            # Count chunks co-mentioning both entities
            n_rows = db.run(
                f"MATCH (s:{src_label} {{name: $src}})-[:MENTIONED_IN]->(c:Chunk)<-[:MENTIONED_IN]-(t:{tgt_label} {{name: $tgt}}) "
                "RETURN count(DISTINCT c) AS n",
                src=edge["src"], tgt=edge["tgt"],
            )
            n = n_rows[0]["n"] if n_rows else k
            n = max(n, k)  # k can't exceed n

            lower, upper = wilson_ci(k, n)
            db.run(
                f"MATCH ()-[r:{rel_type}]->() WHERE id(r)=$rid "
                "SET r.confidence_lower=$lower, r.confidence_upper=$upper",
                rid=edge["rid"], lower=lower, upper=upper,
            )
            updated += 1
    return updated


_CALIBRATION_SYSTEM = """\
You are a calibration assistant for a biomedical question-answering system.
Given a question, an answer, and the source passages used to generate it,
estimate the reliability of the answer as a confidence interval.

Think step by step:
1. How many source passages directly support the answer?
2. Are there any contradictions or caveats in the sources?
3. How complete is the evidence for the specific claims made?

Return ONLY valid JSON (no markdown) with these exact keys:
{
  "score": <integer 0-100>,
  "label": "<High|Medium|Low>",
  "lower_bound": <float 0.0-1.0>,
  "upper_bound": <float 0.0-1.0>,
  "rationale": "<1-2 sentence explanation>",
  "evidence_count": <integer>,
  "contradictions": "<brief note on conflicting evidence, or 'none'>"
}

Calibration guide:
- score 80-100 → High: multiple consistent sources, direct evidence
- score 50-79  → Medium: partial evidence, some inference required
- score 0-49   → Low: limited or indirect evidence, high uncertainty
- lower_bound and upper_bound represent the 95% CI on answer correctness probability
"""


def calibrate_answer_confidence(
    question: str,
    answer: str,
    source_texts: list[str],
    model: str | None = None,
) -> dict[str, Any]:
    """Return LLM-elicited confidence interval for a question-answer pair.

    source_texts: list of chunk texts used to generate the answer
    Returns dict with score, label, lower_bound, upper_bound, rationale, etc.
    """
    model = model or AGENT_MODEL
    sources_block = "\n\n".join(
        f"[Source {i+1}]\n{text[:600]}" for i, text in enumerate(source_texts[:8])
    )
    user_msg = f"Question: {question}\n\nAnswer: {answer}\n\nSources:\n{sources_block}"

    try:
        result = _calibrate_anthropic(user_msg, model)
    except Exception:
        result = _calibrate_groq(user_msg)

    return _validate_confidence(result)


def _calibrate_anthropic(user_msg: str, model: str) -> dict[str, Any]:
    import anthropic  # type: ignore
    import json

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=512,
        system=_CALIBRATION_SYSTEM,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = response.content[0].text if response.content else "{}"
    return json.loads(raw)


def _calibrate_groq(user_msg: str) -> dict[str, Any]:
    from groq import Groq  # type: ignore
    import json, os

    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": _CALIBRATION_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or "{}"
    return json.loads(raw)


def logprob_ci(logprobs: list[float], alpha: float = 0.05) -> tuple[float, float]:
    """Compute a CI from per-token log-probabilities (optional, USE_LOGPROB_CI=true).

    Interprets the mean log-prob as a pseudo-probability and derives a
    beta-distribution CI using the method of moments.
    """
    if not USE_LOGPROB_CI or not logprobs:
        return (0.0, 1.0)

    mean_lp = sum(logprobs) / len(logprobs)
    p_hat = min(max(math.exp(mean_lp), 1e-6), 1.0 - 1e-6)
    n_eff = max(len(logprobs), 1)

    # Beta distribution CI via Wilson approximation on p_hat
    return wilson_ci(round(p_hat * n_eff), n_eff, alpha)


def _validate_confidence(raw: dict[str, Any]) -> dict[str, Any]:
    defaults = {
        "score": 50,
        "label": "Medium",
        "lower_bound": 0.3,
        "upper_bound": 0.7,
        "rationale": "Confidence could not be determined.",
        "evidence_count": 0,
        "contradictions": "none",
    }
    result = {**defaults, **raw}
    result["score"] = max(0, min(100, int(result["score"])))
    result["lower_bound"] = max(0.0, min(1.0, float(result["lower_bound"])))
    result["upper_bound"] = max(0.0, min(1.0, float(result["upper_bound"])))
    if result["lower_bound"] > result["upper_bound"]:
        result["lower_bound"], result["upper_bound"] = result["upper_bound"], result["lower_bound"]
    if result["label"] not in ("High", "Medium", "Low"):
        result["label"] = "Medium"
    return result
