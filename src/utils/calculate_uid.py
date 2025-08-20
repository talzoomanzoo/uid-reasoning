import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

def uid_variance(output, model_path):
    # Split the output by "\n\n" to get segments
    segments = output.split("\n\n")
    
    # Filter out empty segments and calculate id_logprob for each
    logprob_values = []
    for segment in segments:
        segment = segment.strip()
        if segment:  # Only process non-empty segments
            logprob = id_logprob(segment, model_path)
            logprob_values.append(logprob)
    
    # Calculate variance of the logprob values
    if len(logprob_values) > 1:
        return np.var(logprob_values)
    else:
        return 0.0

def uid_gini(output):
    return 0

def uid_shannon(output):
    return 0

def id_logprob(step, model_path):
    """
    Calculate the average log probability of tokens in the input step.
    This requires a pre-trained LM to compute token probabilities.
    """
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        
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

def id_entropy(step):
    return 0

def norm_and_comp(step):
    return 0