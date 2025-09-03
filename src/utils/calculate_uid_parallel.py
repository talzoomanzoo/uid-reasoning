
import math
from typing import Dict, List, Tuple, Any

import numpy as np

from vllm import LLM, SamplingParams

from prompts import get_step_splitting_instruction

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

def calculate_uid_metrics(logprobs_list: List[Dict[int, "Logprob"]], text: str, batch_size: int) -> Dict[str, float]:
    """
    Calculate UID operationalizations for a single reasoning trace, following UID_PoC.

    Parameters
    ----------
    logprobs_list : list of dicts
        Each element corresponds to one generated token position.

    Returns
    -------
    Dict[str, float]
        Variance / Gini-Coefficient / Shannon-evenness for:
          - composite UID (equal weights)
          - LP-only (avg logprob per segment)
          - H-only (avg entropy per segment; full distribution when available)
          - D-only (confidence gap: Δ LP_i across segments)
    """ 
    uid_eq, uid_lp, uid_h, uid_d = create_uid_vectors(logprobs_list, text, batch_size)

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
# UID(z) vector construction
# ------------------------------

def segment_logprobs_by_newlines(logprobs_list: List[Dict[int, "Logprob"]], batch_size: int) -> Tuple[List[List[float]], List[List[float]]]:
    """
    Segment logprobs into segments based on "\n\n" boundaries in chosen tokens.
    
    Returns
    -------
    segments_lp: List of logprob lists for each segment
    segments_h: List of entropy lists for each segment
    """
    segments_lp: List[List[float]] = []        # chosen-token logprobs per step
    segments_h: List[List[float]] = []         # full-distribution entropy per step
    current_lp: List[float] = []
    current_h: List[float] = []
    
    model_path = "Qwen/Qwen2.5-0.5B"
    segment_llm = LLM(model=model_path,
                    gpu_memory_utilization=0.25,
                    max_model_len=8192,
                    max_num_seqs=max(4, batch_size),   # allow full batches
                    enforce_eager=True,
                    dtype="bfloat16",
                    tensor_parallel_size=1,
                    swap_space=32,)

    sampling_params = SamplingParams(
        max_tokens=8192,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        repetition_penalty=1.0,
        skip_special_tokens=True,
        include_stop_str_in_output=False,
    )
    
    for question, solution in zip(questions, solutions):
        prompt = get_step_splitting_instruction(question, solution)
        response = segment_llm.generate([prompt], sampling_params)
        response = response.outputs[0].text
        response = response.split("\n\n")
        response = [r.strip() for r in response if r.strip()]
        
        for r in response:
            if r.startswith("(reasoning step"):
                current_lp.append(r)
            elif r.startswith("(end)"):
                segments_lp.append(current_lp)
                current_lp = []
        segments_h.append(current_lp)
        current_lp = []
                
    for token in logprobs_list:
        if not token:
            # empty token; treat as neutral
            chosen = None
            token_logprobs = []
        else:
            # choose the argmax logprob as the generated token (heuristic)
            chosen = max(token.values(), key=lambda o: float(o.logprob))
            token_logprobs = [float(o.logprob) for o in token.values()]

        # avg logprob uses the chosen token
        lp_chosen = float(chosen.logprob) if chosen is not None else 0.0
        current_lp.append(lp_chosen)

        # step entropy uses full (or top-k) distribution, renormalized
        h_token = entropy_from_logprobs(token_logprobs)
        current_h.append(h_token)

        # segment boundary if the chosen token visually equals a paragraph break
        if chosen is not None and ("\n\n" in (chosen.decoded_token)):
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
    
    return segments_lp, segments_h

def create_uid_vectors(logprobs_list: List[Dict[int, "Logprob"]], text: str, batch_size: int) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    Build UID vectors per segment.

    Segmentation rule: split when the chosen token's decoded text contains "\n\n".
    Since we don't have the explicitly chosen token, we assume it is the token with the
    highest log-probability within that step's dictionary (a common case).

    Returns
    -------
    uid_vector_equal, uid_vector_logprob, uid_vector_entropy, uid_vector_confidence_gap
    """
    # 1) Segment steps by "\\n\\n" observed on the chosen token
    segments_lp, segments_h = segment_logprobs_by_newlines(logprobs_list, text, batch_size)

    # 2) Aggregate per-segment statistics
    lp_values = [float(np.mean(seg)) if len(seg) > 0 else 0.0 for seg in segments_lp]
    h_values = [float(np.mean(seg)) if len(seg) > 0 else 0.0 for seg in segments_h]

    # 3) Confidence gaps Δ LP_i (difference between consecutive segment means)
    # Make length match other vectors by prepending 0.
    d_values = [0.0]
    for i in range(1, len(lp_values)):
        d_values.append(lp_values[i] - lp_values[i - 1])

    # # 4) Within-trace z-normalization of each component
    # lp_norm = _zscore(lp_values)
    # h_norm = _zscore(h_values)
    # d_norm = _zscore(d_values)

    # 5) Composite ID_i with EXACT equal weights 1/3
    uid_equal = [(lp_values[i] - h_values[i] + d_values[i]) / 3.0 for i in range(len(lp_values))]

    # Return all four aligned vectors
    return uid_equal, lp_values, h_values, d_values


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



def _minmax(values: List[float]) -> List[float]:
    if len(values) == 0:
        return []
    arr = np.asarray(values, dtype=float)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    denom = max_val - min_val
    if denom > 0.0 and np.isfinite(denom):
        scaled = (arr - min_val) / denom
    else:
        scaled = np.zeros_like(arr)  # all values identical → map to 0
    return [float(v) for v in scaled]

def _nonnegative_mass(vec: List[float], eps: float = 1e-12) -> np.ndarray:
    """
    Consistently produce a nonnegative vector from arbitrary real-valued IDs.
    We use min-shift + epsilon (no clamping), so that:
        r_i = ID_i - min(ID) + eps  >= eps
    This is the same transform used for variance, Shannon-evenness, and Gini.
    """
    if not vec:
        return np.zeros(0, dtype=float)
    arr = np.asarray(vec, dtype=float)
    m = float(np.min(arr))
    return (arr - m) + eps


# ------------------------------
# UID operationalizations
# ------------------------------

def uid_variance(vec: List[float]) -> float:
    if not vec:
        return 0.0
    r = _nonnegative_mass(vec)
    s = np.asarray(_minmax(r), dtype=float)
    return float(np.var(s, ddof=0))


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