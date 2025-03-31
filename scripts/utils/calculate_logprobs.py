import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional
from vllm.sequence import Logprob

@dataclass
class EntropySpike:
    prev_positions: List[int]
    spike_value: float
    previous_tokens: List[str]
    entropy_diff: float

    def to_dict(self):
        """Convert EntropySpike to a dictionary for JSON serialization"""
        return {
            'prev_positions': self.prev_positions,
            'spike_value': float(self.spike_value),  # ensure float type
            'previous_tokens': self.previous_tokens,
            'entropy_diff': float(self.entropy_diff)  # ensure float type
        }

@dataclass
class EntropyDrop:
    latter_positions: List[int]
    drop_value: float
    latter_tokens: List[str]
    entropy_diff: float

    def to_dict(self):
        """Convert EntropySpike to a dictionary for JSON serialization"""
        return {
            'latter_positions': self.latter_positions,
            'drop_value': float(self.drop_value),  # ensure float type
            'latter_tokens': self.latter_tokens,
            'entropy_diff': float(self.entropy_diff)  # ensure float type
        }

def calculate_entropy(log_probs_list: List[Dict[int, 'Logprob']]) -> float:
    """Calculate entropy from a list of dictionaries containing Logprob objects.
    
    Args:
        log_probs_list: List of dictionaries, each containing a single Logprob object
        Example: [{1249: Logprob(logprob=-0.17, rank=1, decoded_token='To')}, ...]
    """
    # Extract logprob values from each dictionary
    log_probs = [list(d.values())[0].logprob for d in log_probs_list]
    # Convert to probabilities
    probs = np.exp(log_probs)
    # Calculate entropy
    entropy = -np.sum(probs * log_probs)
    return entropy

def calculate_entropy_spike(log_probs_sequence: List[Dict[int, 'Logprob']],
                          window_size: int = 10) -> Optional[EntropySpike]:
    """
    Calculate entropy spikes and return the token before significant spikes.
    
    Args:
        log_probs_sequence: List of dictionaries, each containing a single Logprob object
        threshold: Minimum difference in entropy to consider as a spike
        window_size: Size of the sliding window for entropy calculation
    """

    if len(log_probs_sequence) < window_size:
        return None
    
    # Calculate entropy for each window
    entropies = []
    for i in range(len(log_probs_sequence) - window_size + 1):
        window = log_probs_sequence[i:i + window_size]
        entropy = calculate_entropy(window)
        #import pdb; pdb.set_trace()
        entropies.append(entropy)

    # Find entropy spikes
    max_spike = 0
    spike_idx = -1
    
    entropy_diffs = []
    for i in range(1, len(entropies)):
        entropy_diff = entropies[i] - entropies[i-1]
        entropy_diffs.append(entropy_diff)

    mean = np.mean(entropy_diffs)
    std = np.std(entropy_diffs)
    threshold = mean + 1.5 * std

    for i in range(1, len(entropies)):
        entropy_diff = entropies[i] - entropies[i-1]
        if entropy_diff > threshold and entropy_diff > max_spike:
            max_spike = entropy_diff
            spike_idx = i
    
    # If we found a significant spike
    if spike_idx != -1:
        # Get the dictionary and Logprob object before the spike
        prev_dict = log_probs_sequence[spike_idx - 5 : spike_idx - 1]
        
        spike = EntropySpike(
            prev_positions=[list(d.keys())[0] for d in prev_dict],
            spike_value=entropies[spike_idx],
            previous_tokens=[list(d.values())[0].decoded_token for d in prev_dict],
            entropy_diff=max_spike
        )
        
        return spike.to_dict()
    
    return None


def calculate_entropy_drop(log_probs_sequence: List[Dict[int, 'Logprob']],
                           window_size: int = 10) -> Optional[EntropySpike]:
    """
    Calculate entropy drops and return the token before significant drops.
    
    Args:
        log_probs_sequence: List of dictionaries, each containing a single Logprob object
        window_size: Size of the sliding window for entropy calculation
    """

    if len(log_probs_sequence) < window_size:
        return None

    # Step 1: Calculate entropy for each sliding window
    entropies = []
    for i in range(len(log_probs_sequence) - window_size + 1):
        window = log_probs_sequence[i:i + window_size]
        entropy = calculate_entropy(window)
        entropies.append(entropy)

    # Step 2: Calculate entropy differences between consecutive windows
    entropy_diffs = []
    for i in range(1, len(entropies)):
        entropy_diff = entropies[i] - entropies[i - 1]
        entropy_diffs.append(entropy_diff)

    # Step 3: Adaptive threshold for drops (significant negative diffs)
    mean = np.mean(entropy_diffs)
    std = np.std(entropy_diffs)
    threshold = mean - 1.5 * std  # lower than mean

    # Step 4: Find max drop
    max_drop = 0
    drop_idx = -1

    for i in range(1, len(entropies)):
        entropy_diff = entropies[i] - entropies[i - 1]
        if entropy_diff < threshold and abs(entropy_diff) > abs(max_drop):
            max_drop = entropy_diff
            drop_idx = i

    # Step 5: Return metadata for drop
    if drop_idx != -1:
        latter_dict = log_probs_sequence[drop_idx + 1 : drop_idx + 5]

        drop = EntropyDrop(
            latter_positions=[list(d.keys())[0] for d in latter_dict],
            drop_value=entropies[drop_idx],
            latter_tokens=[list(d.values())[0].decoded_token for d in latter_dict],
            entropy_diff=max_drop
        )

        return drop.to_dict()

    return None


# def calculate_entropy_fall(log_probs):
    
# def calculate_perplexity(log_probs):
#     """Calculate perplexity from log probabilities."""
#     entropy = calculate_entropy(log_probs)
#     perplexity = np.exp(entropy)
#     return perplexity