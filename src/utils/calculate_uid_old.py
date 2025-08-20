import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

#this version is way too slow

def uid_variance(output, model_path):
    # Split the output by "\n\n" to get segments
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)
    
    segments = output.split("\n\n")
    
    # Filter out empty segments and calculate id_logprob for each
    logprob_values = []
    entropy_values = []
    confidence_gap_values = []
    uid_vector = []
    id_score = 0
    id_logprob_score = 0
    id_entropy_score = 0
    id_confidence_gap_score = 0
    for segment in segments:
        segment = segment.strip()
        if segment:  # Only process non-empty segments
            logprob = id_logprob(segment, tokenizer, model)
            logprob_values.append(logprob)
            entropy = id_entropy(segment, tokenizer, model)
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
    
    #calculate variance of the logprob values
    if len(logprob_values) > 1:
        id_logprob_score = np.var(logprob_values)
    else:
        id_logprob_score = 0.0
    
    #calculate entropy
    if len(entropy_values) > 1:
        id_entropy_score = np.var(entropy_values)
    else:
        id_entropy_score = 0.0
    
    #calculate variance of confidence gap
    if len(confidence_gap_values) > 1:
        id_confidence_gap_score = np.var(confidence_gap_values)
    else:
        id_confidence_gap_score = 0.0
        
    #calculate weighted logprob, entropy, confidence gap
    id_score = id_logprob_score *0.3 + id_entropy_score *0.3 + id_confidence_gap_score *0.3
    uid_vector.append(id_score)
    return uid_vector
    
def uid_gini(output):
    return 0

def uid_shannon(output):
    return 0

def id_logprob(step, tokenizer, model):
    """
    Calculate the average log probability of tokens in the input step.
    This requires a pre-trained LM to compute token probabilities.
    """
    
    try:
        
        inputs = tokenizer(step, return_tensors="pt", padding=True, truncation=True)
        
        #get logits from the model
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        #calculate logprobs
        log_probs = torch.log_softmax(logits, dim = -1)
        
        #get the actual token IDs and their corresponding logprobs
        token_ids = inputs['input_ids'][0]
        token_log_probs = []
        
        for i, token_id in enumerate(token_ids[1:], 1):
            log_prob = log_probs[0, i-1, token_id].item()
            token_log_probs.append(log_prob)
            
        #calculate average logprob
        if token_log_probs:
            avg_log_prob = np.mean(token_log_probs)
            return avg_log_prob
        else:
            return 0.0
    
    except Exception as e:  
        print(f"Error in id_logprob: {e}")
    return 0

def id_entropy(step, tokenizer, model):
    """
    Calculate the mean predictive entropy (in nats) over the tokens in `step`.
    This uses next-token prediction: H_t = -sum_v p(v|x_{<t}) log p(v|x_{<t}).
    """
    try:
        inputs = tokenizer(step, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab]

        log_probs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab]
        probs = torch.exp(log_probs)

        # Entropy per position for next-token prediction
        ent_per_pos = -(probs * log_probs).sum(dim=-1)[0]  # [seq_len]

        # Exclude last position (no next token to predict)
        ent_per_pos = ent_per_pos[:-1]

        # Keep only positions where the next token is not padding
        if "attention_mask" in inputs:
            valid_next = inputs["attention_mask"][0, 1:].bool()
            ent_per_pos = ent_per_pos[valid_next]

        if ent_per_pos.numel() == 0:
            return 0.0
        return float(ent_per_pos.mean().item())
    except Exception as e:
        print(f"Error in id_entropy: {e}")
        return 0.0

def id_confidence_gap(curr_logprob, prev_logprob):
    """
    Confidence gap between current segment and previous segment.
    For the first segment, prev_logprob should be 0.0
    """
    return float(curr_logprob - prev_logprob)