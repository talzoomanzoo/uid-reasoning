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

@dataclass
class EntropySpikeAndFall:
    middle_positions: List[int]
    spike_value: float
    middle_tokens: List[str]
    entropy_diff_spike: float
    entropy_diff_fall: float

    def to_dict(self):
        """Convert EntropySpikeAndFall to a dictionary for JSON serialization"""
        return {
            'middle_positions': self.middle_positions,
            'spike_value': float(self.spike_value),  # ensure float type
            'middle_tokens': self.middle_tokens,
            'entropy_diff_spike': float(self.entropy_diff_spike),  # ensure float type
            'entropy_diff_fall': float(self.entropy_diff_fall)  # ensure float type
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

def calculate_entropy_spike_and_fall(log_probs_sequence: List[Dict[int, 'Logprob']],
                                     window_size: int = 142) -> Optional[EntropySpikeAndFall]:
    """
    Calculate entropy spikes and drops and return the 5 tokens in the middle at the point where entropy spikes and then falls.
    
    Args:
        log_probs_sequence: List of dictionaries, each containing a single Logprob object
        window_size: Size of the sliding window for entropy calculation
    """
    if len(log_probs_sequence) < window_size:
        return None
    
    # Calculate entropy for each window
    entropies = []
    for i in range(len(log_probs_sequence) - window_size + 1):
        window = log_probs_sequence[i:i + window_size]
        entropy = calculate_entropy(window)
        entropies.append(entropy)

    # Find entropy differences between consecutive windows
    entropy_diffs = []
    for i in range(1, len(entropies)):
        entropy_diff = entropies[i] - entropies[i-1]
        entropy_diffs.append(entropy_diff)

    # Calculate thresholds for spikes and falls
    mean = np.mean(entropy_diffs)
    std = np.std(entropy_diffs)
    spike_threshold = mean + 1.5 * std
    fall_threshold = mean - 1.5 * std

    # Find windows with both spike and fall
    best_spike_fall_idx = -1
    best_spike_fall_score = 0
    
    for i in range(1, len(entropies) - 1):
        # Check if there's a spike followed by a fall
        if entropy_diffs[i] > spike_threshold and entropy_diffs[i+1] < fall_threshold:
            # Calculate a score based on the magnitude of spike and fall
            score = entropy_diffs[i] - entropy_diffs[i+1]
            if score > best_spike_fall_score:
                best_spike_fall_score = score
                best_spike_fall_idx = i
    
    # If we found a significant spike and fall
    if best_spike_fall_idx != -1:
        # Get the 5 tokens in the middle of the window
        middle_start = best_spike_fall_idx + window_size // 2 - 2
        middle_end = middle_start + 5
        
        # Ensure we don't go out of bounds
        if middle_start < 0:
            middle_start = 0
            middle_end = 5
        elif middle_end > len(log_probs_sequence):
            middle_end = len(log_probs_sequence)
            middle_start = max(0, middle_end - 5)
        
        middle_dict = log_probs_sequence[middle_start:middle_end]
        
        spike_fall = EntropySpikeAndFall(
            middle_positions=list(range(middle_start, middle_end)),
            middle_value=entropies[best_spike_fall_idx],
            middle_tokens=[list(d.values())[0].decoded_token for d in middle_dict],
            entropy_diff_spike=entropy_diffs[best_spike_fall_idx],
            entropy_diff_fall=entropy_diffs[best_spike_fall_idx + 1]
        )
        
        return spike_fall.to_dict()
    
    return None

def calculate_entropy_spike(log_probs_sequence: List[Dict[int, 'Logprob']],
                          window_size: int = 19) -> Optional[EntropySpike]:
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
        prev_dict = log_probs_sequence[spike_idx - 6 : spike_idx - 1]
        
        spike = EntropySpike(
            prev_positions= list(range(spike_idx - 6, spike_idx - 1)),
            spike_value=entropies[spike_idx],
            previous_tokens=[list(d.values())[0].decoded_token for d in prev_dict],
            entropy_diff=max_spike
        )
        
        return spike.to_dict()
    
    return None


def calculate_entropy_drop(log_probs_sequence: List[Dict[int, 'Logprob']],
                           window_size: int = 38) -> Optional[EntropySpike]:
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
        latter_dict = log_probs_sequence[drop_idx + 1 : drop_idx + 6]

        drop = EntropyDrop(
            latter_positions= list(range(drop_idx + 1, drop_idx + 6)),
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