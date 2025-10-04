import numpy as np
from utils.calculate_uid_rev_viz_baselines import calculate_confidence, calculate_entropy, calculate_highest_confidence, calculate_lowest_entropy
from utils.calculate_uid_rev_viz_self_certainty import calculate_self_certainty, calculate_borda_voting_self_certainty
from utils.calculate_uid_rev_viz_cot_decoding import calculate_cot_decoding, calculate_highest_cot_decoding

def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    import itertools

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array(
        [estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)]
    )


def compute_metrics_from_results(results, k_list=[1, 5]):
    total = []
    correct = []
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            gen = np.array(generation)
            all_correct.append(np.all(gen > 0))
        task_ids.append(task_id)
        total.append(len(all_correct))
        correct.append(sum(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist()
        for k in ks
        if (total >= k).all()
    }
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean()
        for k in ks
        if (total >= k).all()
    }
    detail_metrics = {k: dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    pass_at_k["detail"] = detail_metrics
    return pass_at_k


def extract_instance_results(results):
    instance_wise_grades = {}
    for task_id, res in results.items():
        instance_wise_grades[task_id] = []
        for generation in res:
            instance_wise_grades[task_id].append(all([g > 0 for g in generation]))

    instance_wise_grades = [
        v for _, v in sorted(instance_wise_grades.items(), key=lambda item: item[0])
    ]
    return instance_wise_grades

# Calculate proxy pass@k using the actual proxy metrics
def calculate_proxy_pass_at_k_with_actual_metrics(filtered_data, difficulties, k_list):
    """
    Calculate pass@k using actual proxy metrics from the evaluation.
    """
    proxy_pass_at_k = {}
    
    for k in k_list:
        proxy_pass_at_k[f'pass@{k}'] = {}
        
        for difficulty in set(difficulties):
            # Get indices for this difficulty
            difficulty_indices = [i for i, d in enumerate(difficulties) if d == difficulty]
            
            if len(difficulty_indices) == 0:
                continue
                
            # For each question in this difficulty, get the proxy scores
            question_scores = {
                'self_certainty': [],
                'cot_decoding': [],
                'confidence': [],
                'entropy': []
            }
            
            for idx in difficulty_indices:
                item = filtered_data[idx]
                
                # Extract proxy scores for this question
                # Assuming these are stored in the item after evaluation
                if 'borda_voting_self_cert' in item:
                    question_scores['self_certainty'].append(
                        1.0 if item['borda_voting_self_cert'].get('math_equal', False) else 0.0
                    )
                
                if 'highest_cot_decoding' in item:
                    question_scores['cot_decoding'].append(
                        1.0 if item['highest_cot_decoding'].get('math_equal', False) else 0.0
                    )
                
                if 'highest_confidence' in item:
                    question_scores['confidence'].append(
                        1.0 if item['highest_confidence'].get('math_equal', False) else 0.0
                    )
                
                if 'lowest_entropy' in item:
                    question_scores['entropy'].append(
                        1.0 if item['lowest_entropy'].get('math_equal', False) else 0.0
                    )
            
            # Calculate pass@k for each proxy metric
            for proxy_type in ['self_certainty', 'cot_decoding', 'confidence', 'entropy']:
                if proxy_type not in proxy_pass_at_k:
                    proxy_pass_at_k[proxy_type] = {}
                
                if f'pass@{k}' not in proxy_pass_at_k[proxy_type]:
                    proxy_pass_at_k[proxy_type][f'pass@{k}'] = {}
                
                if len(question_scores[proxy_type]) >= k:
                    # Sort by score (descending) and take top k
                    sorted_scores = sorted(question_scores[proxy_type], reverse=True)
                    top_k_scores = sorted_scores[:k]
                    proxy_pass_at_k[proxy_type][f'pass@{k}'][difficulty] = 1.0 if any(score > 0.5 for score in top_k_scores) else 0.0
                else:
                    proxy_pass_at_k[proxy_type][f'pass@{k}'][difficulty] = 0.0
    
    return proxy_pass_at_k
