from typing import List
import numpy as np

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