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

def calculate_cot_decoding(
    logprobs_list: List[Dict[int, "Logprob"]]
) -> Dict[str, float]:
    """
    Calculate confidence score by averaging the difference between the top two 
    logprobability scores at each step.
    
    Args:
        logprobs_list: List of dictionaries mapping token_id -> Logprob for each position
        
    Returns:
        Dictionary containing the confidence score
    """
    if not logprobs_list:
        return {"confidence_score": float("nan")}
    
    step_differences = []
    
    for step_idx, logprobs_dict in enumerate(logprobs_list):
        if not logprobs_dict:
            continue
            
        # Get all logprob values and sort them in descending order
        logprob_values = [logprob.logprob for logprob in logprobs_dict.values()]
        logprob_values.sort(reverse=True)
        
        # Calculate difference between top two logprobs
        if len(logprob_values) >= 2:
            top_diff = logprob_values[0] - logprob_values[1]
            step_differences.append(top_diff)
        elif len(logprob_values) == 1:
            # If only one token, difference is 0 (no uncertainty)
            step_differences.append(0.0)
    
    if not step_differences:
        return {"confidence_score": float("nan")}
    
    # Average the differences across all steps
    confidence_score = sum(step_differences) / len(step_differences)
    
    return {"confidence_score": confidence_score}


def calculate_highest_cot_decoding(per_output_cot_decoding_summary):
    """
    Select the output with the highest confidence score (cot-decoding).
    
    Parameters
    ----------
    per_output_cot_decoding_summary : List[Dict]
        List like:
            [{'output_0': score0, 'math_equal': True}, {'output_1': score1, 'math_equal': False}, ...]
        Each dict should contain exactly one key starting with 'output_' whose value is the confidence score,
        plus 'math_equal' boolean.

    Returns
    -------
    Dict
        The selected dictionary in the form: {"output_{idx}": score, "math_equal": bool}
        If input is empty or all scores invalid, returns {}.
    """
    if not per_output_cot_decoding_summary:
        return {}

    # Extract entries as (output_key, score, math_equal)
    entries = []
    for d in per_output_cot_decoding_summary:
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

    # Select the entry with the highest confidence score
    # Tie-break by math_equal=True, then lower idx
    def tiebreak_key(entry):
        out_key, score, meq = entry
        # tie-break tuple: (score, meq_as_int, negative_idx)
        # Extract numeric idx if present
        try:
            idx = int(out_key.split('_')[-1])
        except Exception:
            idx = 10**9
        return (score, 1 if meq else 0, -idx)

    winner_entry = max(entries, key=tiebreak_key)
    winner_key, winner_score, winner_meq = winner_entry

    return {winner_key: winner_score, "math_equal": winner_meq}

