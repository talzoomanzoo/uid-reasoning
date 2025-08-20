import numpy as np
from vllm.sequence import Logprob
from typing import Dict, List, Tuple

def calculate_uid_metrics(logprobs_list : List[Dict[int, 'Logprob']]) -> Dict[str, float]:
    """
    calculate the metrics of the uid_vector
    """
    uid_vector_equal, uid_vector_logprob, uid_vector_entropy, uid_vector_confidence_gap = create_uid_vectors(logprobs_list)
    uid_variance_equal_score = uid_variance(uid_vector_equal)
    uid_gini_equal_score = uid_gini(uid_vector_equal)
    uid_shannon_equal_score = uid_shannon(uid_vector_equal)
    uid_variance_logprob_score = uid_variance(uid_vector_logprob)
    uid_gini_logprob_score = uid_gini(uid_vector_logprob)
    uid_shannon_logprob_score = uid_shannon(uid_vector_logprob)
    uid_variance_entropy_score = uid_variance(uid_vector_entropy)
    uid_gini_entropy_score = uid_gini(uid_vector_entropy)
    uid_shannon_entropy_score = uid_shannon(uid_vector_entropy)
    uid_variance_confidence_gap_score = uid_variance(uid_vector_confidence_gap)
    uid_gini_confidence_gap_score = uid_gini(uid_vector_confidence_gap)
    uid_shannon_confidence_gap_score = uid_shannon(uid_vector_confidence_gap)
    
    return {
        'uid_variance_equal': uid_variance_equal_score,
        'uid_gini_equal': uid_gini_equal_score,
        'uid_shannon_equal': uid_shannon_equal_score,
        'uid_variance_logprob': uid_variance_logprob_score,
        'uid_gini_logprob': uid_gini_logprob_score,
        'uid_shannon_logprob': uid_shannon_logprob_score,
        'uid_variance_entropy': uid_variance_entropy_score,
        'uid_gini_entropy': uid_gini_entropy_score,
        'uid_shannon_entropy': uid_shannon_entropy_score,
        'uid_variance_confidence_gap': uid_variance_confidence_gap_score,   
        'uid_gini_confidence_gap': uid_gini_confidence_gap_score,
        'uid_shannon_confidence_gap': uid_shannon_confidence_gap_score
    }

def create_uid_vectors(logprobs_list : List[Dict[int, 'Logprob']]) -> Tuple[List[float], List[float], List[float], List[float]]:
    """
    calculate the variance of ID_i across the uid_vector
    """
    
    #segement logprobs by '\n\n' tokens
    segments = []
    current_segment = []
    
    for logprob_dict in logprobs_list:
        for logprob_obj in logprob_dict.values():
            current_segment.append(logprob_obj.logprob)
            if '\n\n' in logprob_obj.decoded_token:
                segments.append(current_segment)
                current_segment = []
    if current_segment:
        segments.append(current_segment)
        
    logprob_values = [] #logprob values of UID vector
    entropy_values = [] #entropy values of UID vector
    confidence_gap_values = [] #confidence gap values of UID vector
    uid_vector = [] #normalized and composite values of UID vector
    uid_vector_logprob = [] #UID vector with only logprob values
    uid_vector_entropy = [] #UID vector with only entropy values
    uid_vector_confidence_gap = [] #UID vector with only confidence gap values
    
    for segment in segments:
        if segment: #only process non-empty segments
            logprob = id_logprob(segment) #representative logprob of the segment
            logprob_values.append(logprob)
            entropy = id_entropy(segment) #representative entropy of the segment
            entropy_values.append(entropy)
            prev_logprob = logprob_values[-2] if len(logprob_values) >= 2 else 0.0
            confidence_gap = id_confidence_gap(logprob, prev_logprob)
            confidence_gap_values.append(confidence_gap)
            
    if len(logprob_values) >= 1:
        mean = float(np.mean(logprob_values))
        std = float(np.std(logprob_values))
        if std > 0:
            logprob_values = [(x - mean) / std for x in logprob_values]
        else:
            logprob_values = [0.0] * len(logprob_values)
            
    if len(entropy_values) >= 1:
        mean = float(np.mean(entropy_values))
        std = float(np.std(entropy_values))
        if std > 0:
            entropy_values = [(x - mean) / std for x in entropy_values]
        else:
            entropy_values = [0.0] * len(entropy_values)
            
    if len(confidence_gap_values) >= 1:
        mean = float(np.mean(confidence_gap_values))
        std = float(np.std(confidence_gap_values))
        if std > 0:
            confidence_gap_values = [(x - mean) / std for x in confidence_gap_values]
        else:
            confidence_gap_values = [0.0] * len(confidence_gap_values)
            
    #calculate the uid_vector
    for i in range(len(logprob_values)):
        uid_vector.append(logprob_values[i] * 0.33 + entropy_values[i] * 0.33 + confidence_gap_values[i] * 0.33)
        uid_vector_logprob.append(logprob_values[i])
        uid_vector_entropy.append(entropy_values[i])
        uid_vector_confidence_gap.append(confidence_gap_values[i])
        
    return uid_vector, uid_vector_logprob, uid_vector_entropy, uid_vector_confidence_gap

def uid_variance(uid_vector : List[float]) -> float:
    return np.var(uid_vector)

def uid_gini(uid_vector : List[float]) -> float:
    if not uid_vector or len(uid_vector) < 2:
        return 0.0

    x = np.asarray(uid_vector, dtype=float)

    # Shift to non-negative if needed
    min_x = np.min(x)
    if min_x < 0:
        x = x - min_x

    total = np.sum(x)
    if total <= 1e-12:
        return 0.0

    x_sorted = np.sort(x)
    n = len(x_sorted)
    indices = np.arange(1, n + 1, dtype=float)

    gini = (2.0 * np.sum(indices * x_sorted) - (n + 1.0) * total) / (n * total)
    return float(max(0.0, min(1.0, gini)))

def uid_shannon(uid_vector : List[float]) -> float:
    if not uid_vector or len(uid_vector) < 2:
        return 0.0
    
    #convert to numpy array and ensure non-negative values
    values = np.array(uid_vector)
    values = np.maximum(values, 0) #ensure non-negative
    
    #normalize to create a probability distribution
    total = np.sum(values)
    if total == 0:
        return 0.0
    probs = values / total
    
    #calculate shannon entropy H = -sum(p*log(p))
    #remove zero probabilities to avoid log(0)
    non_zero_probs = probs[probs > 0]
    if len(non_zero_probs) == 0:
        return 0.0
    
    shannon_entropy = -np.sum(non_zero_probs * np.log(non_zero_probs))
    
    #calculate maximum possible entropy H_max = log(n)
    max_entropy = np.log(len(non_zero_probs))
    
    #Pielou's evenness index J = H / H_max
    if max_entropy == 0:
        return 0.0
    
    evenness = shannon_entropy / max_entropy
    return float(evenness)

def id_logprob(segment):
    #calculate the average logprob of the segment
    avg_logprob = np.mean(segment)
    return avg_logprob

def id_entropy(segment):
    #calculate the mean entropy from a vector of log probabilities
    if not segment:
        return 0.0
    logprobs = np.asarray(segment, dtype=float)
    probs = np.exp(logprobs)
    ent = -probs * logprobs
    return float(np.mean(ent))

def id_confidence_gap(logprob, prev_logprob):
    return logprob - prev_logprob