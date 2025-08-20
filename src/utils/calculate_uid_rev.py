
import math
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    # type hint only; not required at runtime
    from vllm.sequence import Logprob  # noqa: F401
except Exception:  # pragma: no cover - allow running without vLLM installed
    class Logprob:  # minimal stub for type checkers / docs
        def __init__(self, logprob: float, decoded_token: str = "", rank: int = -1) -> None:
            self.logprob = logprob
            self.decoded_token = decoded_token
            self.rank = rank


# ------------------------------
# Public API
# ------------------------------

def calculate_uid_metrics(logprobs_list: List[Dict[int, "Logprob"]]) -> Dict[str, float]:
    """
    Calculate UID operationalizations for a single reasoning trace, following UID_PoC.

    Parameters
    ----------
    logprobs_list : list of dicts
        Each element corresponds to one generated step (token position).
        It should be a mapping {token_id -> Logprob} for *that* position.
        The mapping is ideally the full distribution; if only top-k are present,
        we renormalize the provided subset to approximate H (entropy).

    Returns
    -------
    Dict[str, float]
        Variance / Gini / Shannon-evenness for:
          - composite UID (equal weights)
          - LP-only (avg logprob per segment)
          - H-only (avg entropy per segment; full distribution when available)
          - D-only (confidence gap: Δ LP_i across segments)
    """
    uid_eq, uid_lp, uid_h, uid_d = create_uid_vectors(logprobs_list)

    return {
        # composite
        "uid_variance_equal": uid_variance(uid_eq),
        "uid_gini_equal": uid_gini(uid_eq),
        "uid_shannon_equal": uid_shannon(uid_eq),

        # LP-only
        "uid_variance_logprob": uid_variance(uid_lp),
        "uid_gini_logprob": uid_gini(uid_lp),
        "uid_shannon_logprob": uid_shannon(uid_lp),

        # H-only
        "uid_variance_entropy": uid_variance(uid_h),
        "uid_gini_entropy": uid_gini(uid_h),
        "uid_shannon_entropy": uid_shannon(uid_h),

        # D-only
        "uid_variance_confidence_gap": uid_variance(uid_d),
        "uid_gini_confidence_gap": uid_gini(uid_d),
        "uid_shannon_confidence_gap": uid_shannon(uid_d),
    }


# ------------------------------
# Vector construction
# ------------------------------

def create_uid_vectors(logprobs_list: List[Dict[int, "Logprob"]]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Build UID vectors per segment.

    Segmentation rule: split when the *chosen* token's decoded text contains "\\n\\n".
    Since we don't have the explicit chosen token, we assume it is the token with the
    highest log-probability within that step's dictionary (a common case).

    Returns
    -------
    uid_vector_equal, uid_vector_logprob, uid_vector_entropy, uid_vector_confidence_gap
    """
    # 1) Segment steps by "\\n\\n" observed on the chosen token
    segments_lp: List[List[float]] = []        # chosen-token logprobs per step
    segments_h: List[List[float]] = []         # full-distribution entropy per step
    current_lp: List[float] = []
    current_h: List[float] = []

    for step in logprobs_list:
        if not step:
            # empty step; treat as neutral
            chosen = None
            step_logprobs = []
        else:
            # choose the argmax logprob as the generated token (heuristic)
            chosen = max(step.values(), key=lambda o: float(o.logprob))
            step_logprobs = [float(o.logprob) for o in step.values()]

        # avg logprob uses the chosen token
        lp_chosen = float(chosen.logprob) if chosen is not None else 0.0
        current_lp.append(lp_chosen)

        # step entropy uses full (or top-k) distribution, renormalized
        h_step = entropy_from_logprobs(step_logprobs)
        current_h.append(h_step)

        # segment boundary if the chosen token visually equals a paragraph break
        if chosen is not None and ("\n\n" in (chosen.decoded_token or "")):
            segments_lp.append(current_lp)
            segments_h.append(current_h)
            current_lp, current_h = [], []

    # flush trailing
    if current_lp:
        segments_lp.append(current_lp)
        segments_h.append(current_h)

    # Edge case: if no segments were found, create a single segment
    if not segments_lp:
        segments_lp = [current_lp or [0.0]]
        segments_h = [current_h or [0.0]]

    # 2) Aggregate per-segment statistics
    lp_values = [float(np.mean(seg)) if len(seg) > 0 else 0.0 for seg in segments_lp]
    h_values = [float(np.mean(seg)) if len(seg) > 0 else 0.0 for seg in segments_h]

    # 3) Confidence gaps Δ LP_i (difference between consecutive segment means)
    # Make length match other vectors by prepending 0.
    d_values = [0.0]
    for i in range(1, len(lp_values)):
        d_values.append(lp_values[i] - lp_values[i - 1])

    # 4) Within-trace z-normalization of each component
    lp_norm = _zscore(lp_values)
    h_norm = _zscore(h_values)
    d_norm = _zscore(d_values)

    # 5) Composite ID_i with EXACT equal weights 1/3
    uid_equal = [(lp_norm[i] + h_norm[i] + d_norm[i]) / 3.0 for i in range(len(lp_norm))]

    # Return all four aligned vectors
    return uid_equal, lp_norm, h_norm, d_norm


# ------------------------------
# Component computations
# ------------------------------

def entropy_from_logprobs(logprobs: List[float]) -> float:
    """
    Compute Shannon entropy H = -sum p log p from (possibly partial) log-probabilities.
    If only top-k candidates are provided, renormalize exp(logprob) over the subset.

    Uses natural logarithms; units are nats.
    """
    if not logprobs:
        return 0.0
    x = np.asarray(logprobs, dtype=float)
    # subtract max for numerical stability
    m = float(np.max(x))
    exps = np.exp(x - m)
    Z = float(np.sum(exps))
    if Z <= 0.0 or not np.isfinite(Z):
        return 0.0
    p = exps / Z
    # avoid log(0); mask zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.log(p, where=p > 0.0, out=np.full_like(p, -np.inf))
    h = -np.sum(p * logp)
    return float(h)


def _zscore(values: List[float]) -> List[float]:
    if not values:
        return []
    arr = np.asarray(values, dtype=float)
    mu = float(np.mean(arr))
    sigma = float(np.std(arr))
    if sigma > 0.0 and np.isfinite(sigma):
        z = (arr - mu) / sigma
    else:
        z = np.zeros_like(arr)
    return [float(v) for v in z]


# ------------------------------
# UID operationalizations
# ------------------------------

def uid_variance(vec: List[float]) -> float:
    """Population variance of ID_i across segments (translation-invariant)."""
    if not vec:
        return 0.0
    return float(np.var(np.asarray(vec, dtype=float)))


def _nonnegative_mass(vec: List[float], eps: float = 1e-12) -> np.ndarray:
    """
    Consistently produce a nonnegative vector from arbitrary real-valued IDs.
    We use min-shift + epsilon (no clamping), so that:
        r_i = ID_i - min(ID) + eps  >= eps
    This is the same transform used for both Shannon-evenness and Gini.
    """
    if not vec:
        return np.zeros(0, dtype=float)
    arr = np.asarray(vec, dtype=float)
    m = float(np.min(arr))
    return (arr - m) + eps


def uid_shannon(vec: List[float]) -> float:
    """
    Shannon evenness of the per-segment IDs.

    Steps (per UID_PoC):
      1) Convert IDs to a probability distribution q_i over segments.
         We use r_i = ID_i - min(ID) + eps to ensure non-negativity without clamping.
      2) H = -sum q_i log q_i
      3) Evenness = H / log(n)

    Returns a number in [0, 1] for n >= 2; returns 0.0 if n < 2 or sum r_i == 0.
    """
    r = _nonnegative_mass(vec)
    n = int(r.size)
    if n <= 1:
        return 0.0
    total = float(np.sum(r))
    if total <= 0.0 or not np.isfinite(total):
        return 0.0
    q = r / total
    with np.errstate(divide="ignore", invalid="ignore"):
        logq = np.log(q, where=q > 0.0, out=np.full_like(q, -np.inf))
    H = -float(np.sum(q * logq))
    return float(H / math.log(n)) if n > 1 else 0.0


def uid_gini(vec: List[float]) -> float:
    """
    Gini coefficient on the same nonnegative mass r_i used for evenness.
    Scale-invariant but not translation-invariant, hence using min-shift.
    Returns in [0, 1].
    """
    r = _nonnegative_mass(vec)
    n = int(r.size)
    if n == 0:
        return 0.0
    s = float(np.sum(r))
    if s <= 0.0 or not np.isfinite(s):
        return 0.0
    sorted_r = np.sort(r)
    cum = float(np.sum((np.arange(1, n + 1) * sorted_r)))
    g = (2.0 * cum) / (n * s) - (n + 1.0) / n
    return float(g)


# ------------------------------
# Optional: Evaluation utilities
# ------------------------------

def roc_auc(scores: List[float], labels: List[int]) -> float:
    """
    Compute ROC AUC from scores and binary labels without external deps.
    Larger scores are assumed to indicate the positive class.
    """
    if not scores or len(scores) != len(labels):
        return float("nan")

    # Rank-based (Mann–Whitney U) implementation
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=float)

    pos = np.array(labels, dtype=int) == 1
    n_pos = int(np.sum(pos))
    n_neg = len(scores) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum_pos = float(np.sum(ranks[pos]))
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def log_loss(probs: List[float], labels: List[int], eps: float = 1e-12) -> float:
    """
    Compute logistic log-loss given predicted probabilities for the positive class.
    """
    if not probs or len(probs) != len(labels):
        return float("nan")
    p = np.clip(np.asarray(probs, dtype=float), eps, 1.0 - eps)
    y = np.asarray(labels, dtype=int)
    loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    return float(loss)
