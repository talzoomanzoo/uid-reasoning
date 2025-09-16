
import math
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt

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

def calculate_uid_metrics(logprobs_list: List[Dict[int, "Logprob"]], thinkseg: bool) -> Dict[str, float]:
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
    uid_eq, uid_lp, uid_h, uid_d = create_uid_vectors(logprobs_list, thinkseg)
    id_eq_linear, id_lp_linear, id_h_linear, id_d_linear = create_id_linear(uid_eq, uid_lp, uid_h, uid_d)

    return {
        # composite
        "uid_variance_equal": uid_variance(uid_eq),
        "uid_gini_equal": uid_gini(uid_eq),
        "uid_shannon_equal": uid_shannon(uid_eq),
        "uid_l2_equal": uid_l2(uid_eq, id_eq_linear),

        # LP-only
        "uid_variance_logprob": uid_variance(uid_lp),
        "uid_gini_logprob": uid_gini(uid_lp),
        "uid_shannon_logprob": uid_shannon(uid_lp),
        "uid_l2_logprob": uid_l2(uid_lp, id_lp_linear),
        
        # H-only
        "uid_variance_entropy": uid_variance(uid_h),
        "uid_gini_entropy": uid_gini(uid_h),
        "uid_shannon_entropy": uid_shannon(uid_h),
        "uid_l2_entropy": uid_l2(uid_h, id_h_linear),
        
        # D-only
        "uid_variance_confidence_gap": uid_variance(uid_d),
        "uid_gini_confidence_gap": uid_gini(uid_d),
        "uid_shannon_confidence_gap": uid_shannon(uid_d),
        "uid_l2_confidence_gap": uid_l2(uid_d, id_d_linear),
    }


# ------------------------------
# UID(z) vector construction
# ------------------------------

def create_uid_vectors(logprobs_list: List[Dict[int, "Logprob"]], thinkseg: bool) -> Tuple[List[float], List[float], List[float], List[float]]:
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
    segments_lp, segments_h = segment_by_paragraph_breaks(logprobs_list, thinkseg)

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

    # 6) Apply non-negative mass transformation to all vectors
    uid_equal_nonneg = _nonnegative_mass(uid_equal).tolist()
    uid_lp_nonneg = _nonnegative_mass(lp_values).tolist()
    uid_h_nonneg = _nonnegative_mass(h_values).tolist()
    uid_d_nonneg = _nonnegative_mass(d_values).tolist()

    # Return all four aligned vectors (now non-negative)
    return uid_equal_nonneg, uid_lp_nonneg, uid_h_nonneg, uid_d_nonneg


def create_id_linear(uid_eq: List[float], uid_lp: List[float], uid_h: List[float], uid_d: List[float]) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fit a linear function y = a*x + b to each UID vector.

    Parameters
    ----------
    uid_eq, uid_lp, uid_h, uid_d : List[float]
        Per-segment UID vectors (already non-negative from create_uid_vectors).

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]
        {
          "equal": {"slope": a, "intercept": b, "fitted": List[float]},
          "logprob": {"slope": a, "intercept": b, "fitted": List[float]},
          "entropy": {"slope": a, "intercept": b, "fitted": List[float]},
          "confidence_gap": {"slope": a, "intercept": b, "fitted": List[float]},
        }
    """
    def _fit_line(y: List[float]) -> Dict[str, Any]:
        n = len(y)
        if n == 0:
            return {"slope": 0.0, "intercept": 0.0, "fitted": []}
        if n == 1:
            b = float(y[0])
            return {"slope": 0.0, "intercept": b, "fitted": [b]}
        x = np.arange(n, dtype=float)
        a, b = np.polyfit(x, np.asarray(y, dtype=float), 1)
        yhat = (a * x + b).astype(float).tolist()
        return {"slope": float(a), "intercept": float(b), "fitted": yhat}

    return _fit_line(uid_eq), _fit_line(uid_lp), _fit_line(uid_h), _fit_line(uid_d)


def segment_by_paragraph_breaks(logprobs_list, thinkseg):
    """
    Segment logprobs into segments based on paragraph breaks (\n\n).
    If thinkseg=True, only use segments that appear before the </think> token.
    Returns lists of segments for logprobs and entropies.
    """
    segments_lp = []
    segments_h = []
    current_lp = []
    current_h = []
    found_think_end = False
    
    for token in logprobs_list:
        if not token:
            # empty token; treat as neutral
            chosen = None
            token_logprobs = []
        else:
            # choose the argmax logprob as the generated token (heuristic)
            chosen = max(token.values(), key=lambda o: float(o.logprob))
            token_logprobs = [float(o.logprob) for o in token.values()]

        # Check for </think> token if thinkseg is enabled
        if thinkseg and chosen is not None and "</think>" in chosen.decoded_token:
            found_think_end = True
            # Still process this token but mark that we've found the end

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
            
            # If we've found </think> and thinkseg is enabled, stop processing
            if thinkseg and found_think_end:
                break
    
    # Add the last segment if it's not empty and we haven't found </think>
    if current_lp and not (thinkseg and found_think_end):
        segments_lp.append(current_lp)
        segments_h.append(current_h)
    
    return segments_lp, segments_h


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
    # vec is already non-negative from create_uid_vectors
    s = np.asarray(_minmax(vec), dtype=float)
    return float(np.var(s, ddof=0))


def uid_shannon(vec: List[float]) -> float:
    """
    Shannon evenness of the per-segment IDs.

    Steps (per UID_PoC):
      1) Convert IDs to a probability distribution q_i over segments.
         vec is already non-negative from create_uid_vectors
      2) H = -sum q_i log q_i
      3) Evenness = H / log(n)

    Returns a number in [0, 1] for n >= 2; returns 0.0 if n < 2 or sum r_i == 0.
    """
    r = np.asarray(vec, dtype=float)  # vec is already non-negative
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
    vec is already non-negative from create_uid_vectors.
    Returns in [0, 1].
    """
    r = np.asarray(vec, dtype=float)  # vec is already non-negative
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


def visualize_id_vectors(uid_equal, uid_lp, uid_h, uid_d, dataset_name, model_path, split, title="ID Scores Across Steps", save_path=None):
    """
    Visualize the ID vectors across steps in a line plot.
    
    Parameters
    ----------
    uid_equal : List[float]
        Composite ID vector with equal weights
    uid_lp : List[float]
        Log probability vector
    uid_h : List[float]
        Entropy vector
    uid_d : List[float]
        Confidence gap vector
    dataset_name : str
        Name of the dataset
    model_path : str
        Path to the model
    split : str
        Dataset split
    title : str, optional
        Title for the plot
    save_path : str, optional
        Path to save the plot. If None, saves to default location.
    """
    import os
    from datetime import datetime
    
    # Create segment indices
    segments = list(range(len(uid_equal)))
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot each vector
    plt.plot(segments, uid_equal, 'b-', linewidth=2, label='Composite ID (Equal Weights)', marker='o')
    plt.plot(segments, uid_lp, 'r-', linewidth=2, label='Log Probability', marker='s')
    plt.plot(segments, uid_h, 'g-', linewidth=2, label='Entropy', marker='^')
    plt.plot(segments, uid_d, 'm-', linewidth=2, label='Confidence Gap', marker='d')
    
    # Customize the plot
    plt.xlabel('Step Index', fontsize=12)
    plt.ylabel('ID Score', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Add horizontal line at y=0 for reference
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    if save_path is None:
        # Default save path with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/id_scores_across_steps_{dataset_name}.{split}.{model_path}_{timestamp}.png"
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {save_path}")
    
    plt.close()

# Add this to the existing calculate_uid_metrics function to return vectors for visualization
def calculate_id_metrics_with_vectors(logprobs_list: List[Dict[int, "Logprob"]], thinkseg: bool) -> Tuple[Dict[str, float], List[float], List[float], List[float], List[float]]:
    """
    Calculate ID metrics and return both metrics and vectors for visualization.
    
    Returns
    -------
    Tuple containing:
    - metrics_dict: Dictionary of ID metrics
    - uid_equal: Composite ID vector
    - uid_lp: Log probability vector  
    - uid_h: Entropy vector
    - uid_d: Confidence gap vector
    """
    uid_eq, uid_lp, uid_h, uid_d = create_uid_vectors(logprobs_list, thinkseg)
    id_eq_linear, id_lp_linear, id_h_linear, id_d_linear = create_id_linear(uid_eq, uid_lp, uid_h, uid_d)
    metrics_dict = {
        # composite
        "uid_variance_equal": uid_variance(uid_eq),
        "uid_gini_equal": uid_gini(uid_eq),
        "uid_shannon_equal": uid_shannon(uid_eq),
        "uid_l2_equal": uid_l2(uid_eq, id_eq_linear),


        # LP-only
        "uid_variance_logprob": uid_variance(uid_lp),
        "uid_gini_logprob": uid_gini(uid_lp),
        "uid_shannon_logprob": uid_shannon(uid_lp),
        "uid_l2_logprob": uid_l2(uid_lp, id_lp_linear),

        # H-only
        "uid_variance_entropy": uid_variance(uid_h),
        "uid_gini_entropy": uid_gini(uid_h),
        "uid_shannon_entropy": uid_shannon(uid_h),
        "uid_l2_entropy": uid_l2(uid_h, id_h_linear),
        # D-only
        "uid_variance_confidence_gap": uid_variance(uid_d),
        "uid_gini_confidence_gap": uid_gini(uid_d),
        "uid_shannon_confidence_gap": uid_shannon(uid_d),
        "uid_l2_confidence_gap": uid_l2(uid_d, id_d_linear),
    }
    
    return metrics_dict, uid_eq, uid_lp, uid_h, uid_d

def visualize_average_step_counts(data, dataset_name, model_path, split, thinkseg=False):
    """
    Visualize average step counts across all samples.
    For math500, also visualize by level.
    For all datasets, also visualize by math_equal (correct/incorrect).
    
    Parameters
    ----------
    data : List[Dict]
        List of data items with ID metrics
    dataset_name : str
        Name of the dataset
    model_path : str
        Path to the model
    split : str
        Dataset split
    thinkseg : bool
        Whether to use thinkseg
    """
    import os
    from datetime import datetime
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Collect all step counts (vector lengths)
    all_step_counts = []
    
    # For math500, also collect by level
    level_step_counts = {} if dataset_name == 'math500' else None
    domain_step_counts = {} if dataset_name == 'gpqa' else None
    
    # For all datasets, collect by math_equal
    math_equal_step_counts = {'True': [], 'False': []}
    
    for item in data:
        # Find all ID metrics for this item
        idx = 0
        while f"id_metrics_{idx}_metrics" in item:
            metrics_key = f"id_metrics_{idx}_metrics"
            if metrics_key in item:
                # Get the vector length (number of steps/segments)
                uid_eq = item.get(f"id_equal_{idx}", [])
                if uid_eq:  # If vector exists and is not empty
                    step_count = len(uid_eq)  # This is the actual number of steps
                    all_step_counts.append(step_count)
                    
                    # Get math_equal status for this output
                    math_equal = item.get(f"Metrics_{idx}", {}).get("math_equal", False)
                    math_equal_key = str(math_equal)
                    math_equal_step_counts[math_equal_key].append(step_count)
                    
                    # For math500, group by level
                    if dataset_name == 'math500' and level_step_counts is not None:
                        level = item.get("level", "Unknown")
                        if level not in level_step_counts:
                            level_step_counts[level] = []
                        level_step_counts[level].append(step_count)
                    if dataset_name == 'gpqa' and domain_step_counts is not None:
                        domain = item.get("High-level domain", "Unknown")
                        if domain not in domain_step_counts:
                            domain_step_counts[domain] = []
                        domain_step_counts[domain].append(step_count)
            idx += 1
    
    if not all_step_counts:
        print("No step counts found to visualize")
        return
    
    # Calculate average step count
    avg_step_count = np.mean(all_step_counts)
    std_step_count = np.std(all_step_counts)
    
    # Create overall step count visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Overall average step count
    plt.figure(figsize=(10, 6))
    plt.bar(['Average Steps'], [avg_step_count], color='skyblue', alpha=0.7)
    plt.errorbar(['Average Steps'], [avg_step_count], yerr=[std_step_count], 
                fmt='none', color='red', capsize=5, capthick=2)
    plt.ylabel('Number of Steps', fontsize=12)
    plt.title(f'Average Step Count - {dataset_name} {split}', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add text annotation
    plt.text(0, avg_step_count + std_step_count + 0.5, 
             f'Mean: {avg_step_count:.2f} ± {std_step_count:.2f}', 
             ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Save overall plot
    overall_save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/avg_step_count_{dataset_name}.{split}.{model_path}_{timestamp}.png"
    os.makedirs(os.path.dirname(overall_save_path), exist_ok=True)
    plt.savefig(overall_save_path, dpi=300, bbox_inches='tight')
    print(f"Overall step count plot saved to: {overall_save_path}")
    plt.close()
    
    # Plot 2: Step counts by math_equal (correct/incorrect)
    if math_equal_step_counts['True'] or math_equal_step_counts['False']:
        math_equal_labels = []
        math_equal_means = []
        math_equal_stds = []
        
        if math_equal_step_counts['True']:
            math_equal_labels.append('Correct')
            math_equal_means.append(np.mean(math_equal_step_counts['True']))
            math_equal_stds.append(np.std(math_equal_step_counts['True']))
        
        if math_equal_step_counts['False']:
            math_equal_labels.append('Incorrect')
            math_equal_means.append(np.mean(math_equal_step_counts['False']))
            math_equal_stds.append(np.std(math_equal_step_counts['False']))
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(math_equal_labels, math_equal_means, color=['lightgreen', 'lightcoral'], alpha=0.7)
        plt.errorbar(math_equal_labels, math_equal_means, yerr=math_equal_stds, 
                    fmt='none', color='red', capsize=5, capthick=2)
        
        # Add value labels on bars
        for i, (bar, avg, std) in enumerate(zip(bars, math_equal_means, math_equal_stds)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1, 
                    f'{avg:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xlabel('Answer Correctness', fontsize=12)
        plt.ylabel('Average Number of Steps', fontsize=12)
        plt.title(f'Average Step Count by Correctness - {dataset_name} {split}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save math_equal plot
        math_equal_save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/avg_step_count_by_correctness_{dataset_name}.{split}.{model_path}_{timestamp}.png"
        os.makedirs(os.path.dirname(math_equal_save_path), exist_ok=True)
        plt.savefig(math_equal_save_path, dpi=300, bbox_inches='tight')
        print(f"Correctness step count plot saved to: {math_equal_save_path}")
        plt.close()
    
    # For math500, also visualize by level
    if dataset_name == 'math500' and level_step_counts:
        # Plot 3: Step counts by level
        levels = sorted(level_step_counts.keys())
        level_avg_counts = [np.mean(level_step_counts[level]) for level in levels]
        level_std_counts = [np.std(level_step_counts[level]) for level in levels]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(levels, level_avg_counts, color='lightcoral', alpha=0.7)
        plt.errorbar(levels, level_avg_counts, yerr=level_std_counts, 
                    fmt='none', color='red', capsize=5, capthick=2)
        
        # Add value labels on bars
        for i, (bar, avg, std) in enumerate(zip(bars, level_avg_counts, level_std_counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1, 
                    f'{avg:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Level', fontsize=12)
        plt.ylabel('Average Number of Steps', fontsize=12)
        plt.title(f'Average Step Count by Level - {dataset_name} {split}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save level plot
        level_save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/avg_step_count_by_level_{dataset_name}.{split}.{model_path}_{timestamp}.png"
        os.makedirs(os.path.dirname(level_save_path), exist_ok=True)
        plt.savefig(level_save_path, dpi=300, bbox_inches='tight')
        print(f"Level step count plot saved to: {level_save_path}")
        plt.close()
        
        # Print summary statistics
        print(f"\nStep Count Summary for {dataset_name} {split}:")
        print(f"Overall: {avg_step_count:.2f} ± {std_step_count:.2f} steps")
        print("By Level:")
        for level in levels:
            avg = np.mean(level_step_counts[level])
            std = np.std(level_step_counts[level])
            count = len(level_step_counts[level])
            print(f"  Level {level}: {avg:.2f} ± {std:.2f} steps (n={count})")
    elif dataset_name == 'gpqa' and domain_step_counts:
        # Plot 3: Step counts by domain
        domains = sorted(domain_step_counts.keys())
        domain_avg_counts = [np.mean(domain_step_counts[domain]) for domain in domains]
        domain_std_counts = [np.std(domain_step_counts[domain]) for domain in domains]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(domains, domain_avg_counts, color='lightcoral', alpha=0.7)
        plt.errorbar(domains, domain_avg_counts, yerr=domain_std_counts, 
                    fmt='none', color='red', capsize=5, capthick=2)
        
        # Add value labels on bars
        for i, (bar, avg, std) in enumerate(zip(bars, domain_avg_counts, domain_std_counts)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.1, 
                    f'{avg:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=9)
        
        plt.xlabel('Domain', fontsize=12)
        plt.ylabel('Average Number of Steps', fontsize=12)
        plt.title(f'Average Step Count by Domain - {dataset_name} {split}', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save domain plot
        domain_save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/avg_step_count_by_domain_{dataset_name}.{split}.{model_path}_{timestamp}.png"
        os.makedirs(os.path.dirname(domain_save_path), exist_ok=True)
        plt.savefig(domain_save_path, dpi=300, bbox_inches='tight')
        print(f"Domain step count plot saved to: {domain_save_path}")
        plt.close()
        
        # Print summary statistics
        print(f"\nStep Count Summary for {dataset_name} {split}:")
        print(f"Overall: {avg_step_count:.2f} ± {std_step_count:.2f} steps")
        print("By Domain:")
        for domain in domains:
            avg = np.mean(domain_step_counts[domain])
            std = np.std(domain_step_counts[domain])
            count = len(domain_step_counts[domain])
            print(f"  Domain {domain}: {avg:.2f} ± {std:.2f} steps (n={count})")
    else:
        print(f"\nStep Count Summary for {dataset_name} {split}:")
        print(f"Overall: {avg_step_count:.2f} ± {std_step_count:.2f} steps (n={len(all_step_counts)})")
    
    # Print math_equal summary
    if math_equal_step_counts['True'] or math_equal_step_counts['False']:
        print("By Correctness:")
        if math_equal_step_counts['True']:
            avg = np.mean(math_equal_step_counts['True'])
            std = np.std(math_equal_step_counts['True'])
            count = len(math_equal_step_counts['True'])
            print(f"  Correct: {avg:.2f} ± {std:.2f} steps (n={count})")
        if math_equal_step_counts['False']:
            avg = np.mean(math_equal_step_counts['False'])
            std = np.std(math_equal_step_counts['False'])
            count = len(math_equal_step_counts['False'])
            print(f"  Incorrect: {avg:.2f} ± {std:.2f} steps (n={count})")

def visualize_average_id_vectors(data, dataset_name, model_path, split, thinkseg=False, step_limit=200):
    """
    Visualize average ID vectors across all samples.
    For math500, also visualize by level.
    For all datasets, also visualize by math_equal (correct/incorrect).
    
    Parameters
    ----------
    data : List[Dict]
        List of data items with ID metrics
    dataset_name : str
        Name of the dataset
    model_path : str
        Path to the model
    split : str
        Dataset split
    thinkseg : bool
        Whether to use thinkseg
    step_limit : int
        Number of steps to limit
    """
    import os
    from datetime import datetime
    import numpy as np
    
    # Collect all ID vectors
    all_uid_equal = []
    all_uid_lp = []
    all_uid_h = []
    all_uid_d = []
    
    # For math500, also collect by level
    level_vectors = {} if dataset_name == 'math500' else None
    domain_vectors = {} if dataset_name == 'gpqa' else None
    
    # For all datasets, collect by math_equal
    math_equal_vectors = {'True': {'id_equal': [], 'id_lp': [], 'id_h': [], 'id_d': []},
                         'False': {'id_equal': [], 'id_lp': [], 'id_h': [], 'id_d': []}}
    
    for item in data:
        # Find all ID metrics for this item
        idx = 0
        while f"id_metrics_{idx}_metrics" in item:
            metrics_key = f"id_metrics_{idx}_metrics"
            if metrics_key in item:
                # Extract vectors from the stored data
                uid_eq = item.get(f"id_equal_{idx}", [])
                uid_lp = item.get(f"id_lp_{idx}", [])
                uid_h = item.get(f"id_h_{idx}", [])
                uid_d = item.get(f"id_d_{idx}", [])
                
                if uid_eq and uid_lp and uid_h and uid_d:
                    # NEW: Filter by step count (length of vector)
                    step_count = len(uid_eq)
                    if step_count <= step_limit:
                        all_uid_equal.append(uid_eq)
                        all_uid_lp.append(uid_lp)
                        all_uid_h.append(uid_h)
                        all_uid_d.append(uid_d)
                        
                        # Get math_equal status for this output
                        math_equal = item.get(f"Metrics_{idx}", {}).get("math_equal", False)
                        math_equal_key = str(math_equal)
                        math_equal_vectors[math_equal_key]['id_equal'].append(uid_eq)
                        math_equal_vectors[math_equal_key]['id_lp'].append(uid_lp)
                        math_equal_vectors[math_equal_key]['id_h'].append(uid_h)
                        math_equal_vectors[math_equal_key]['id_d'].append(uid_d)
                        
                        # For math500, group by level
                        if dataset_name == 'math500' and level_vectors is not None:
                            level = item.get("level", "Unknown")
                            if level not in level_vectors:
                                level_vectors[level] = {
                                    'id_equal': [], 'id_lp': [], 'id_h': [], 'id_d': []
                                }
                            level_vectors[level]['id_equal'].append(uid_eq)
                            level_vectors[level]['id_lp'].append(uid_lp)
                            level_vectors[level]['id_h'].append(uid_h)
                            level_vectors[level]['id_d'].append(uid_d)
                        if dataset_name == 'gpqa' and domain_vectors is not None:
                            domain = item.get("High-level domain", "Unknown")
                            if domain not in domain_vectors:
                                domain_vectors[domain] = {
                                    'id_equal': [], 'id_lp': [], 'id_h': [], 'id_d': []
                                }
                            domain_vectors[domain]['id_equal'].append(uid_eq)
                            domain_vectors[domain]['id_lp'].append(uid_lp)
                            domain_vectors[domain]['id_h'].append(uid_h)
                            domain_vectors[domain]['id_d'].append(uid_d)
            idx += 1
    
    if not all_uid_equal:
        print(f"No ID vectors found to visualize (step_limit={step_limit})")
        return
    
    print(f"Visualizing {len(all_uid_equal)} vectors with step_count <= {step_limit}")
    
    # Calculate average vectors for all data
    max_length = max(len(vec) for vec in all_uid_equal)
    
    # Pad all vectors to the same length
    padded_uid_equal = []
    padded_uid_lp = []
    padded_uid_h = []
    padded_uid_d = []
    
    for vec in all_uid_equal:
        padded = vec + [0.0] * (max_length - len(vec))
        padded_uid_equal.append(padded)
    
    for vec in all_uid_lp:
        padded = vec + [0.0] * (max_length - len(vec))
        padded_uid_lp.append(padded)
    
    for vec in all_uid_h:
        padded = vec + [0.0] * (max_length - len(vec))
        padded_uid_h.append(padded)
    
    for vec in all_uid_d:
        padded = vec + [0.0] * (max_length - len(vec))
        padded_uid_d.append(padded)
    
    # Calculate averages
    avg_uid_equal = np.mean(padded_uid_equal, axis=0)
    avg_uid_lp = np.mean(padded_uid_lp, axis=0)
    avg_uid_h = np.mean(padded_uid_h, axis=0)
    avg_uid_d = np.mean(padded_uid_d, axis=0)
    
    # Visualize overall average
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/avg_id_scores_{dataset_name}.{split}.{model_path}_{timestamp}.png"
    
    visualize_id_vectors(
        avg_uid_equal.tolist(), avg_uid_lp.tolist(), avg_uid_h.tolist(), avg_uid_d.tolist(),
        dataset_name, model_path, split,
        title=f"Average ID Scores Across Steps - {dataset_name} {split}",
        save_path=save_path
    )
    
    # Visualize by math_equal (correct/incorrect)
    for correctness, vectors in math_equal_vectors.items():
        if not vectors['id_equal']:
            continue
            
        # Calculate average for this correctness group
        correctness_max_length = max(len(vec) for vec in vectors['id_equal'])
        
        # Pad vectors for this correctness group
        correctness_padded_equal = []
        correctness_padded_lp = []
        correctness_padded_h = []
        correctness_padded_d = []
        
        for vec in vectors['id_equal']:
            padded = vec + [0.0] * (correctness_max_length - len(vec))
            correctness_padded_equal.append(padded)
        
        for vec in vectors['id_lp']:
            padded = vec + [0.0] * (correctness_max_length - len(vec))
            correctness_padded_lp.append(padded)
        
        for vec in vectors['id_h']:
            padded = vec + [0.0] * (correctness_max_length - len(vec))
            correctness_padded_h.append(padded)
        
        for vec in vectors['id_d']:
            padded = vec + [0.0] * (correctness_max_length - len(vec))
            correctness_padded_d.append(padded)
        
        # Calculate correctness averages
        correctness_avg_equal = np.mean(correctness_padded_equal, axis=0)
        correctness_avg_lp = np.mean(correctness_padded_lp, axis=0)
        correctness_avg_h = np.mean(correctness_padded_h, axis=0)
        correctness_avg_d = np.mean(correctness_padded_d, axis=0)
        
        # Visualize correctness average
        correctness_label = "Correct" if correctness == "True" else "Incorrect"
        correctness_save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/avg_id_scores_{dataset_name}.{split}.{correctness_label.lower()}.{model_path}_{timestamp}.png"
        
        visualize_id_vectors(
            correctness_avg_equal.tolist(), correctness_avg_lp.tolist(), correctness_avg_h.tolist(), correctness_avg_d.tolist(),
            dataset_name, model_path, split,
            title=f"Average ID Scores Across Steps - {dataset_name} {split} ({correctness_label})",
            save_path=correctness_save_path
        )
    
    if dataset_name == 'gpqa' and domain_vectors:
        for domain, vectors in domain_vectors.items():
            if not vectors['id_equal']:
                continue
                
            # Calculate average for this domain
            domain_max_length = max(len(vec) for vec in vectors['id_equal'])
            
            # Pad vectors for this domain
            domain_padded_equal = []
            domain_padded_lp = []
            domain_padded_h = []
            domain_padded_d = []
            
            for vec in vectors['id_equal']:
                padded = vec + [0.0] * (domain_max_length - len(vec))
                domain_padded_equal.append(padded)
            
            for vec in vectors['id_lp']:
                padded = vec + [0.0] * (domain_max_length - len(vec))
                domain_padded_lp.append(padded)
            
            for vec in vectors['id_h']:
                padded = vec + [0.0] * (domain_max_length - len(vec))
                domain_padded_h.append(padded)
            
            for vec in vectors['id_d']:
                padded = vec + [0.0] * (domain_max_length - len(vec))
                domain_padded_d.append(padded)
            
            # Calculate domain averages
            domain_avg_equal = np.mean(domain_padded_equal, axis=0)
            domain_avg_lp = np.mean(domain_padded_lp, axis=0)
            domain_avg_h = np.mean(domain_padded_h, axis=0)
            domain_avg_d = np.mean(domain_padded_d, axis=0)
            
            # Visualize domain average
            domain_save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/avg_id_scores_{dataset_name}.{split}.domain_{domain}.{model_path}_{timestamp}.png"
            
            visualize_id_vectors(
                domain_avg_equal.tolist(), domain_avg_lp.tolist(), domain_avg_h.tolist(), domain_avg_d.tolist(),
                dataset_name, model_path, split,
                title=f"Average ID Scores Across Steps - {dataset_name} {split} Domain {domain}",
                save_path=domain_save_path
            )
    # For math500, also visualize by level
    elif dataset_name == 'math500' and level_vectors:
        for level, vectors in level_vectors.items():
            if not vectors['id_equal']:
                continue
                
            # Calculate average for this level
            level_max_length = max(len(vec) for vec in vectors['id_equal'])
            
            # Pad vectors for this level
            level_padded_equal = []
            level_padded_lp = []
            level_padded_h = []
            level_padded_d = []
            
            for vec in vectors['id_equal']:
                padded = vec + [0.0] * (level_max_length - len(vec))
                level_padded_equal.append(padded)
            
            for vec in vectors['id_lp']:
                padded = vec + [0.0] * (level_max_length - len(vec))
                level_padded_lp.append(padded)
            
            for vec in vectors['id_h']:
                padded = vec + [0.0] * (level_max_length - len(vec))
                level_padded_h.append(padded)
            
            for vec in vectors['id_d']:
                padded = vec + [0.0] * (level_max_length - len(vec))
                level_padded_d.append(padded)
            
            # Calculate level averages
            level_avg_equal = np.mean(level_padded_equal, axis=0)
            level_avg_lp = np.mean(level_padded_lp, axis=0)
            level_avg_h = np.mean(level_padded_h, axis=0)
            level_avg_d = np.mean(level_padded_d, axis=0)
            
            # Visualize level average
            level_save_path = f"/scratch2/mjgwak/uid-reasoning/scripts/outputs/runs.baselines/avg_id_scores_{dataset_name}.{split}.level_{level}.{model_path}_{timestamp}.png"
            
            visualize_id_vectors(
                level_avg_equal.tolist(), level_avg_lp.tolist(), level_avg_h.tolist(), level_avg_d.tolist(),
                dataset_name, model_path, split,
                title=f"Average ID Scores Across Steps - {dataset_name} {split} Level {level}",
                save_path=level_save_path
            )

def uid_l2(
    y: List[float],
    fit: Dict[str, Any],
    alpha: float = 1.5,
    tail_k: int = 10,
    min_score: float = 0.0,
    max_score: float = 1.0,
    eps: float = 1e-12,
) -> float:
    """
    Weighted L2 distance between a UID vector and its fitted line.
    Optionally min-max normalize the final score with provided bounds.

    Returns
    -------
    float
        raw_score                  if min_score or max_score is None/invalid
        (raw_score - min)/(max-min) clipped to [0,1] otherwise
    """
    y_arr = np.asarray(y, dtype=float)
    yhat = np.asarray(fit.get("fitted", []), dtype=float)
    if y_arr.size == 0 or yhat.size == 0:
        raw = 0.0
    else:
        n = min(y_arr.size, yhat.size)
        r = y_arr[:n] - yhat[:n]
        w = np.ones(n, dtype=float)
        if tail_k > 0 and alpha != 1.5:
            k = min(tail_k, n)
            w[-k:] *= float(alpha)
        raw = float(np.sqrt(np.sum(w * (r ** 2))))

    if min_score is None or max_score is None or not np.isfinite(min_score) or not np.isfinite(max_score) or (max_score - min_score) <= eps:
        return raw

    x = (raw - float(min_score)) / (float(max_score) - float(min_score))
    return float(np.clip(x, 0.0, 1.0))