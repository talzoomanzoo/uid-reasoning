import json
import argparse
import os
import numpy as np
from collections import defaultdict


def analyze_uid_accuracy(data):
    """
    Analyze accuracy based on UID scores.
    For each problem, find outputs with highest and lowest UID scores for each metric,
    then calculate their accuracy.
    """
    
    # Define UID metrics to analyze
    uid_metrics = [
        "uid_variance_equal", "uid_gini_equal", "uid_shannon_equal",
        "uid_variance_logprob", "uid_gini_logprob", "uid_shannon_logprob", 
        "uid_variance_entropy", "uid_gini_entropy", "uid_shannon_entropy",
        "uid_variance_confidence_gap", "uid_gini_confidence_gap", "uid_shannon_confidence_gap"
    ]
    
    results = {
        'highest_uid': defaultdict(list),
        'lowest_uid': defaultdict(list)
    }
    
    for problem in data:
        problem_id = problem.get('id', 'unknown')
        
        # Find all outputs for this problem
        outputs = []
        i = 0
        while f'Output_{i}' in problem:
            output_key = f'Output_{i}'
            metrics_key = f'Metrics_{i}'
            uid_metrics_key = f'uid_metrics_{i}'
            
            if uid_metrics_key in problem and metrics_key in problem:
                outputs.append({
                    'index': i,
                    'uid_metrics': problem[uid_metrics_key],
                    'accuracy': problem[metrics_key].get('acc', 0),
                    'exact_match': problem[metrics_key].get('em', 0),
                    'f1': problem[metrics_key].get('f1', 0)
                })
            i += 1
        
        if not outputs:
            continue
            
        # Analyze each UID metric
        for metric in uid_metrics:
            # Find output with highest UID score
            highest_output = max(outputs, key=lambda x: x['uid_metrics'].get(metric, -float('inf')))
            highest_uid_score = highest_output['uid_metrics'].get(metric, 0)
            
            # Find output with lowest UID score
            lowest_output = min(outputs, key=lambda x: x['uid_metrics'].get(metric, float('inf')))
            lowest_uid_score = lowest_output['uid_metrics'].get(metric, 0)
            
            # Store results
            results['highest_uid'][metric].append({
                'problem_id': problem_id,
                'output_index': highest_output['index'],
                'uid_score': highest_uid_score,
                'accuracy': highest_output['accuracy'],
                'exact_match': highest_output['exact_match'],
                'f1': highest_output['f1']
            })
            
            results['lowest_uid'][metric].append({
                'problem_id': problem_id,
                'output_index': lowest_output['index'],
                'uid_score': lowest_uid_score,
                'accuracy': lowest_output['accuracy'],
                'exact_match': lowest_output['exact_match'],
                'f1': lowest_output['f1']
            })
    
    return results


def calculate_summary_stats(results):
    """Calculate summary statistics for the results."""
    summary = {}
    
    for uid_type in ['highest_uid', 'lowest_uid']:
        summary[uid_type] = {}
        
        for metric in results[uid_type]:
            accuracies = [item['accuracy'] for item in results[uid_type][metric]]
            exact_matches = [item['exact_match'] for item in results[uid_type][metric]]
            f1_scores = [item['f1'] for item in results[uid_type][metric]]
            uid_scores = [item['uid_score'] for item in results[uid_type][metric]]
            
            summary[uid_type][metric] = {
                'count': len(accuracies),
                'mean_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'mean_exact_match': np.mean(exact_matches),
                'std_exact_match': np.std(exact_matches),
                'mean_f1': np.mean(f1_scores),
                'std_f1': np.std(f1_scores),
                'mean_uid_score': np.mean(uid_scores),
                'std_uid_score': np.std(uid_scores)
            }
    
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", default="analysis_out")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.input_file, "r") as f:
        data = json.load(f)
    
    print(f"Analyzing {len(data)} problems...")
    
    # Analyze UID-based accuracy
    results = analyze_uid_accuracy(data)
    
    # Calculate summary statistics
    summary = calculate_summary_stats(results)
    
    # Print results
    print("\n" + "="*80)
    print("ACCURACY ANALYSIS BASED ON UID SCORES")
    print("="*80)
    
    for uid_type in ['highest_uid', 'lowest_uid']:
        print(f"\n{uid_type.upper().replace('_', ' ')} UID SCORES:")
        print("-" * 60)
        
        for metric in sorted(summary[uid_type].keys()):
            stats = summary[uid_type][metric]
            print(f"{metric:35} | Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} | "
                  f"Exact Match: {stats['mean_exact_match']:.4f} ± {stats['std_exact_match']:.4f} | "
                  f"F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f} | "
                  f"UID Score: {stats['mean_uid_score']:.4f} ± {stats['std_uid_score']:.4f}")
    
    # Save detailed results
    results_file = os.path.join(args.outdir, "uid_accuracy_analysis.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(args.outdir, "uid_accuracy_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()