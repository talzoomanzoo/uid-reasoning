from collections import Counter
from typing import Dict, List, Any, Tuple


def calculate_majority_voting(per_output_majority_voting_summary):
    """
    Perform majority voting by finding the most frequent majority_voting_obj.
    
    Parameters
    ----------
    per_output_majority_voting_summary : List[Dict]
        List like:
            [{'output_0': (output_text, labeled_answer), 'math_equal': True}, 
             {'output_1': (output_text, labeled_answer), 'math_equal': False}, ...]
        Each dict should contain exactly one key starting with 'output_' whose value is the majority_voting_obj tuple,
        plus 'math_equal' boolean.
    
    Returns
    -------
    Dict
        The selected dictionary in the form: {"majority_voting_obj": tuple, "math_equal": bool}
        If input is empty, returns {}.
    """
    if not per_output_majority_voting_summary:
        return {}
    
    # Extract all majority_voting_obj entries with their math_equal values
    obj_entries = []
    for d in per_output_majority_voting_summary:
        # Find the output_* key
        out_keys = [k for k in d.keys() if k.startswith('output_')]
        if not out_keys:
            continue
        out_key = out_keys[0]
        majority_voting_obj = d.get(out_key, None)
        math_equal = bool(d.get('math_equal', False))
        
        # Skip invalid entries
        if majority_voting_obj is None:
            continue
        
        obj_entries.append((majority_voting_obj, math_equal))
    
    if not obj_entries:
        return {}
    
    # Count frequency of each majority_voting_obj
    obj_counter = Counter(obj for obj, _ in obj_entries)
    
    if not obj_counter:
        return {}
    
    # Find the most frequent majority_voting_obj
    most_frequent_obj, count = obj_counter.most_common(1)[0]
    
    # If there are ties, prefer the one with math_equal=True
    # Count math_equal values for the most frequent object
    math_equal_counter = Counter(
        math_eq for obj, math_eq in obj_entries 
        if obj == most_frequent_obj
    )
    
    # Get the most common math_equal value (prefer True if tied)
    most_common_math_equal = math_equal_counter.most_common(1)[0][0]
    if len(math_equal_counter) > 1 and math_equal_counter[True] == math_equal_counter[False]:
        most_common_math_equal = True
    
    return {
        "majority_voting_obj": most_frequent_obj,
        "math_equal": most_common_math_equal
    }