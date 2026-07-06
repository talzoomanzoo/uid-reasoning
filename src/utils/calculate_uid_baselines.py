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



def calculate_confidence(
    logprobs_list: List[Dict[int, "Logprob"]]
) -> Dict[str, float]:
    """
    Calculate the average logprobability of all tokens across all positions.
    
    Args:
        logprobs_list: List of dictionaries mapping token_id -> Logprob for each position
        
    Returns:
        Dictionary containing the average logprobability
    """
    if not logprobs_list:
        return {"calculate_confidence": float("nan")}
    
    all_logprobs = []
    
    for step_idx, logprobs_dict in enumerate(logprobs_list):
        if not logprobs_dict:
            continue
            
        # Get all logprob values for this step
        step_logprobs = [logprob.logprob for logprob in logprobs_dict.values()]
        all_logprobs.extend(step_logprobs)
    
    if not all_logprobs:
        return {"calculate_confidence": float("nan")}
    
    # Calculate the average logprobability across all tokens
    average_logprob = sum(all_logprobs) / len(all_logprobs)
    
    return {"calculate_confidence": average_logprob}

def calculate_highest_confidence(per_output_confidence_summary):
    """
    Select the output with the highest confidence score.
    """
    if not per_output_confidence_summary:
        return {}
    
    entries = []
    for d in per_output_confidence_summary:
        out_keys = [k for k in d.keys() if k.startswith('output_')]
        if not out_keys:
            continue
        out_key = out_keys[0]
        score = d.get(out_key, None)
        meq = bool(d.get('math_equal', False))

        if score is None or (isinstance(score, float) and (math.isnan(score) or not math.isfinite(score))):
            continue
            
        entries.append((out_key, float(score), meq))
        
    if not entries:
        return {}

    def tiebreak_key(entry):
        out_key, score, meq = entry

        try:
            idx = int(out_key.split('_')[-1])
        except Exception:
            idx = 10**9
        return (score, 1 if meq else 0, -idx)
    
    winner_entry = max(entries, key=tiebreak_key)
    winner_key, winner_score, winner_meq = winner_entry
    return {winner_key: winner_score, "math_equal": winner_meq}


def calculate_entropy(
    logprobs_list: List[Dict[int, "Logprob"]]
) -> Dict[str, float]:
    """
    Calculate the average entropy of all tokens across all positions.
    
    Entropy is calculated as: H = -sum(p_i * log(p_i)) where p_i is the probability of token i
    
    Args:
        logprobs_list: List of dictionaries mapping token_id -> Logprob for each position
        
    Returns:
        Dictionary containing the average entropy
    """
    if not logprobs_list:
        return {"calculate_entropy": float("nan")}
    
    step_entropies = []
    
    for step_idx, logprobs_dict in enumerate(logprobs_list):
        if not logprobs_dict:
            continue
            
        # Convert logprobs to probabilities
        logprob_values = [logprob.logprob for logprob in logprobs_dict.values()]
        
        # Convert log probabilities to probabilities
        # p_i = exp(logprob_i) / sum(exp(logprob_j) for all j)
        max_logprob = max(logprob_values)  # For numerical stability
        exp_logprobs = [math.exp(lp - max_logprob) for lp in logprob_values]
        sum_exp = sum(exp_logprobs)
        
        if sum_exp == 0:
            continue
            
        probabilities = [exp_lp / sum_exp for exp_lp in exp_logprobs]
        
        # Calculate entropy for this step: H = -sum(p_i * log(p_i))
        step_entropy = 0.0
        for prob in probabilities:
            step_entropy -= prob * math.log(prob)
        step_entropies.append(step_entropy)
    
    if not step_entropies:
        return {"calculate_entropy": float("nan")}
    
    # Calculate the average entropy across all steps
    average_entropy = sum(step_entropies) / len(step_entropies)
    
    return {"calculate_entropy": average_entropy}


def calculate_lowest_entropy(per_output_entropy_summary):
    """
    Select the output with the highest entropy score.
    """
    if not per_output_entropy_summary:
        return {}
    
    entries = []
    for d in per_output_entropy_summary:
        out_keys = [k for k in d.keys() if k.startswith('output_')]
        if not out_keys:
            continue
        out_key = out_keys[0]
        score = d.get(out_key, None)
        meq = bool(d.get('math_equal', False))

        if score is None or (isinstance(score, float) and (math.isnan(score) or not math.isfinite(score))):
            continue
            
        entries.append((out_key, float(score), meq))

    if not entries:
        return {}

    def tiebreak_key(entry):
        out_key, score, meq = entry

        try:
            idx = int(out_key.split('_')[-1])
        except Exception:
            idx = 10**9
        return (score, 1 if meq else 0, -idx)
    
    winner_entry = min(entries, key=tiebreak_key)
    winner_key, winner_score, winner_meq = winner_entry
    return {winner_key: winner_score, "math_equal": winner_meq}