import math
from typing import Dict, List, Optional, Any

try:
    from vllm.sequence import Logprob  # noqa: F401
except Exception:  # pragma: no cover
    class Logprob:
        def __init__(self, logprob: float, decoded_token: str = "", rank: int = -1) -> None:
            self.logprob = logprob
            self.decoded_token = decoded_token
            self.rank = rank


def calculate_self_certainty(
    logprobs_list: List[Dict[int, "Logprob"]],
    vocab_size: Optional[int] = None,
    eps: float = 1e-12,
) -> Dict[str, float]:
    """
    Compute self-certainty with the paper's formulation:

        Self-certainty = - (1 / (n * V)) * sum_{i=1..n} sum_{j=1..V} log(V * p(j | x, y_<i))

    vLLM logprobs often provide only top-k entries per position. We therefore:
      - convert provided logprobs to probs,
      - compute the remaining mass (1 - sum(top-k)),
      - distribute it uniformly over the (V - k) missing tokens,
      - and sum logs over all V tokens.

    Args:
        logprobs_list: list (length n) of dicts for each position i.
                       Each dict maps token_id -> Logprob where Logprob.logprob is ln p_j.
        vocab_size: total vocabulary size V. REQUIRED for faithful computation if
                    the per-position dicts are top-k (varying size).
        eps: numeric floor to avoid log(0) due to rounding.

    Returns:
        {"self_certainty": float}
    """
    if not logprobs_list:
        return {"self_certainty": float("nan")}

    n = len(logprobs_list)

    # If V not supplied, try to infer. If any position has different k, we still need V.
    # We'll fall back to the max observed k, but warn via docstring: this is a *lower bound*
    # (it ignores true tail mass across the unseen tokens).
    if vocab_size is None:
        # Best-effort fallback (approximate, not exact)
        vocab_size = max(len(d) for d in logprobs_list)
        # You may want to pass tokenizer.vocab_size here for faithful results.

    V = int(vocab_size)
    if V <= 0:
        raise ValueError("vocab_size must be a positive integer.")

    logV = math.log(V)
    total_sum = 0.0

    for idx, dist in enumerate(logprobs_list):
        if not dist:
            raise ValueError(f"Empty logprob dictionary at position {idx}.")

        # Convert provided logprobs to probs and sum them
        probs = []
        sum_topk = 0.0
        for lp in dist.values():
            val = getattr(lp, "logprob", None)
            if val is None or not math.isfinite(val):
                raise ValueError(f"Invalid logprob at position {idx}: {val}")
            p = math.exp(val)
            # guard tiny negatives from numeric issues
            if p < 0.0:
                p = 0.0
            probs.append(p)
            sum_topk += p

        # Clip/renormalize minor overshoot due to rounding
        if sum_topk > 1.0:
            # Renormalize the listed tokens to sum to (1 - tiny_epsilon)
            scale = (1.0 - eps) / sum_topk
            probs = [max(eps, p * scale) for p in probs]
            sum_topk = sum(probs)

        k = len(probs)
        missing = max(V - k, 0)

        # Remaining probability mass
        rem = 1.0 - sum_topk
        if rem < 0.0:
            rem = 0.0
        # Distribute uniformly to the (V - k) unseen tokens
        if missing > 0:
            p_tail = max(rem / missing, eps)
        else:
            # No missing slots; if there's leftover mass due to numerical quirks, smear into eps adjustments
            p_tail = None

        # sum_j log(V * p_j) = V*log V + sum_j log p_j over all V tokens
        sum_log_p_full = 0.0

        # Listed tokens
        for p in probs:
            sum_log_p_full += math.log(max(p, eps))

        # Unlisted tokens (uniform completion)
        if missing > 0:
            sum_log_p_full += missing * math.log(p_tail)

        total_sum += (V * logV) + sum_log_p_full

    self_certainty = -(1.0 / (n * V)) * total_sum
    return {"self_certainty": float(self_certainty)}


def calculate_borda_voting_self_certainty(per_output_self_cert_summary, p: float = 1.0):
    """
    Perform Borda-style voting over outputs ranked by self-certainty.

    Parameters
    ----------
    per_output_self_cert_summary : List[Dict]
        List like:
            [{'output_0': score0, 'math_equal': True}, {'output_1': score1, 'math_equal': False}, ...]
        Each dict should contain exactly one key starting with 'output_' whose value is the self-certainty score,
        plus 'math_equal' boolean.

    p : float
        Power in v(r) = (N - r + 1)^p. p=0 → majority (all equal votes). Larger p → top-ranked dominates.
        Here, implement p=0.5

    Returns
    -------
    Dict
        The voted dictionary in the form: {"output_{idx}": score, "math_equal": bool}
        If input is empty or all scores invalid, returns {}.
    """
    import math

    if not per_output_self_cert_summary:
        return {}

    # Extract entries as (output_key, score, math_equal)
    entries = []
    for d in per_output_self_cert_summary:
        # find the output_* key
        out_keys = [k for k in d.keys() if k.startswith('output_')]
        if not out_keys:
            continue
        out_key = out_keys[0]
        score = d.get(out_key, None)
        meq = bool(d.get('math_equal', False))

        # skip invalid scores
        if score is None or (isinstance(score, float) and (math.isnan(score) or not math.isfinite(score))):
            continue

        entries.append((out_key, float(score), meq))

    if not entries:
        return {}

    # Rank by score descending (higher self-certainty → better rank)
    entries_sorted = sorted(entries, key=lambda x: x[1], reverse=True)
    N = len(entries_sorted)

    # Assign votes v(r) = (N - r + 1)^p (r starts at 1)
    votes = {}
    for rank, (out_key, score, meq) in enumerate(entries_sorted, start=1):
        v = (N - rank + 1) ** 0.5 #p=0.5
        votes[out_key] = votes.get(out_key, 0.0) + v

    # Pick the winner by highest votes; tie-break by higher score, then math_equal=True, then lower idx
    def tiebreak_key(out_key):
        # find score, meq for this out_key
        for ok, sc, meq in entries:
            if ok == out_key:
                # tie-break tuple: (votes, score, meq_as_int, negative_idx)
                # Extract numeric idx if present
                try:
                    idx = int(ok.split('_')[-1])
                except Exception:
                    idx = 10**9
                return (votes[out_key], sc, 1 if meq else 0, -idx)
        # fallback
        return (votes[out_key], float('-inf'), 0, 0)

    winner_key = max(votes.keys(), key=tiebreak_key)

    # Retrieve winner score and math_equal
    winner_score = None
    winner_meq = False
    for ok, sc, meq in entries:
        if ok == winner_key:
            winner_score = sc
            winner_meq = meq
            break

    return {winner_key: winner_score, "math_equal": winner_meq} 