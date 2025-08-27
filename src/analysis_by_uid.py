import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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


def calculate_aggregated_stats(summary):
    """Calculate aggregated statistics across all UID metrics for highest_uid and lowest_uid."""
    aggregated = {}
    
    for uid_type in ['highest_uid', 'lowest_uid']:
        mean_accuracies = []
        std_accuracies = []
        mean_exact_matches = []
        std_exact_matches = []
        mean_f1_scores = []
        std_f1_scores = []
        mean_uid_scores = []
        std_uid_scores = []
        
        for metric in summary[uid_type]:
            stats = summary[uid_type][metric]
            mean_accuracies.append(stats['mean_accuracy'])
            std_accuracies.append(stats['std_accuracy'])
            mean_exact_matches.append(stats['mean_exact_match'])
            std_exact_matches.append(stats['std_exact_match'])
            mean_f1_scores.append(stats['mean_f1'])
            std_f1_scores.append(stats['std_f1'])
            mean_uid_scores.append(stats['mean_uid_score'])
            std_uid_scores.append(stats['std_uid_score'])
        
        aggregated[uid_type] = {
            'mean_accuracy': np.mean(mean_accuracies),
            'std_accuracy': np.std(mean_accuracies),
            'mean_accuracy_std': np.mean(std_accuracies),
            'std_accuracy_std': np.std(std_accuracies),
            'mean_exact_match': np.mean(mean_exact_matches),
            'std_exact_match': np.std(mean_exact_matches),
            'mean_exact_match_std': np.mean(std_exact_matches),
            'std_exact_match_std': np.std(std_exact_matches),
            'mean_f1': np.mean(mean_f1_scores),
            'std_f1': np.std(mean_f1_scores),
            'mean_f1_std': np.mean(std_f1_scores),
            'std_f1_std': np.std(std_f1_scores),
            'mean_uid_score': np.mean(mean_uid_scores),
            'std_uid_score': np.std(mean_uid_scores),
            'mean_uid_score_std': np.mean(std_uid_scores),
            'std_uid_score_std': np.std(std_uid_scores)
        }
    
    return aggregated

def create_boxplots(summary, results, outdir, filename_prefix, input_filename=None):
    """Create bar graphs for mean accuracy and boxplots for individual metrics by UID type."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Prepare data
    uid_types = ['highest_uid', 'lowest_uid']
    metrics = list(summary['highest_uid'].keys())
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 1, figsize=(16, 12))
    if input_filename:
        dataset_name = input_filename.split("/")[-2] if "/" in input_filename else input_filename.replace(".json", "")
        fig.suptitle('UID Analysis: Mean Accuracy by Individual Metric \n ' + dataset_name, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('UID Analysis: Mean Accuracy by Individual Metric \n ' + filename_prefix, fontsize=16, fontweight='bold')
    
    # Plot: Bar graph of mean accuracy by individual metric
    ax2 = axes
    metric_labels = [metric.replace('uid_', '').replace('_', ' ').title() for metric in metrics]
    highest_mean_acc = [summary['highest_uid'][metric]['mean_accuracy'] for metric in metrics]
    lowest_mean_acc = [summary['lowest_uid'][metric]['mean_accuracy'] for metric in metrics]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    bars2_highest = ax2.bar(x - width/2, highest_mean_acc, width, label='Highest UID', color='lightblue', alpha=0.8)
    bars2_lowest = ax2.bar(x + width/2, lowest_mean_acc, width, label='Lowest UID', color='lightcoral', alpha=0.8)

    # Add value labels on top of bars
    for bar, value in zip(bars2_highest, highest_mean_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    for bar, value in zip(bars2_lowest, lowest_mean_acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

    ax2.set_title('Mean Accuracy by Individual Metric')
    ax2.set_ylabel('Mean Accuracy')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metric_labels, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Save the plot
    plot_file = os.path.join(outdir, f"{filename_prefix}_uid_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", default="analysis_out")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.input, "r") as f:
        data = json.load(f)
    
    print(f"Analyzing {len(data)} problems...")
    
    # Analyze UID-based accuracy
    results = analyze_uid_accuracy(data)
    
    # Calculate summary statistics
    summary = calculate_summary_stats(results)
    
    # Calculate aggregated statistics
    aggregated = calculate_aggregated_stats(summary)
    
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
    
    # Print aggregated results
    print("\n" + "="*80)
    print("AGGREGATED STATISTICS ACROSS ALL UID METRICS")
    print("="*80)
    
    for uid_type in ['highest_uid', 'lowest_uid']:
        print(f"\n{uid_type.upper().replace('_', ' ')} UID SCORES (AGGREGATED):")
        print("-" * 60)
        stats = aggregated[uid_type]
        print(f"Mean Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f}")
        print(f"Mean Accuracy Std: {stats['mean_accuracy_std']:.4f} ± {stats['std_accuracy_std']:.4f}")
        print(f"Mean Exact Match: {stats['mean_exact_match']:.4f} ± {stats['std_exact_match']:.4f}")
        print(f"Mean Exact Match Std: {stats['mean_exact_match_std']:.4f} ± {stats['std_exact_match_std']:.4f}")
        print(f"Mean F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
        print(f"Mean F1 Std: {stats['mean_f1_std']:.4f} ± {stats['std_f1_std']:.4f}")
        print(f"Mean UID Score: {stats['mean_uid_score']:.4f} ± {stats['std_uid_score']:.4f}")
        print(f"Mean UID Score Std: {stats['mean_uid_score_std']:.4f} ± {stats['std_uid_score_std']:.4f}")
    
    # Create boxplots
    filename_prefix = args.input.split("/")[-2] if "/" in args.input else args.input.replace(".json", "")
    plot_file = create_boxplots(summary, results, args.outdir, filename_prefix, args.input)
    
    # Save detailed results
    results_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_analysis.json")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    
    # Save summary
    summary_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save aggregated results
    aggregated_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_aggregated.json")
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Aggregated results saved to: {aggregated_file}")
    print(f"Boxplots saved to: {plot_file}")


if __name__ == "__main__":
    main()