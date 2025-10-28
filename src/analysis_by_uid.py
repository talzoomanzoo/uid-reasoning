import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
try:
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Correlation analysis will be skipped.")


def analyze_selected_uid_accuracy(data):
    """
    Analyze accuracy for selected traces (highest and lowest uid_variance_entropy).
    This function handles the data structure after trace selection.
    """
    # Define UID metrics to analyze
    uid_metrics = [
        "uid_variance_entropy", "uid_gini_entropy", "uid_shannon_entropy",
    ]
    
    results = {
        'highest_uid': {metric: [] for metric in uid_metrics},
        'lowest_uid': {metric: [] for metric in uid_metrics}
    }
    
    for problem in data:
        problem_id = problem.get('id', 'unknown')
        
        # Check if this problem has selected traces
        if 'Output_highest' in problem and 'id_metrics_highest_metrics' in problem:
            highest_metrics = problem['id_metrics_highest_metrics']
            highest_accuracy = problem['Metrics_highest'].get('acc', 0)
            highest_exact_match = problem['Metrics_highest'].get('em', 0)
            highest_f1 = problem['Metrics_highest'].get('f1', 0)
            
            for metric in uid_metrics:
                uid_score = highest_metrics.get(metric, 0)
                results['highest_uid'][metric].append({
                    'problem_id': problem_id,
                    'output_index': 'highest',
                    'uid_score': uid_score,
                    'accuracy': highest_accuracy,
                    'exact_match': highest_exact_match,
                    'f1': highest_f1
                })
        
        if 'Output_lowest' in problem and 'id_metrics_lowest_metrics' in problem:
            lowest_metrics = problem['id_metrics_lowest_metrics']
            lowest_accuracy = problem['Metrics_lowest'].get('acc', 0)
            lowest_exact_match = problem['Metrics_lowest'].get('em', 0)
            lowest_f1 = problem['Metrics_lowest'].get('f1', 0)
            
            for metric in uid_metrics:
                uid_score = lowest_metrics.get(metric, 0)
                results['lowest_uid'][metric].append({
                    'problem_id': problem_id,
                    'output_index': 'lowest',
                    'uid_score': uid_score,
                    'accuracy': lowest_accuracy,
                    'exact_match': lowest_exact_match,
                    'f1': lowest_f1
                })
    
    return results


def analyze_uid_accuracy(data):
    """
    Analyze accuracy based on UID scores.
    For each problem, find outputs with highest and lowest UID scores for each metric,
    then calculate their accuracy.
    """
    
    # Define UID metrics to analyze
    uid_metrics = [
        # "uid_variance_equal", "uid_gini_equal", "uid_shannon_equal",
        # "uid_variance_logprob", "uid_gini_logprob", "uid_shannon_logprob", 
        "uid_variance_entropy", "uid_gini_entropy", "uid_shannon_entropy",
        # "uid_variance_confidence_gap", "uid_gini_confidence_gap", "uid_shannon_confidence_gap",
        # "uid_l2_equal", "uid_l2_logprob", "uid_l2_entropy", "uid_l2_confidence_gap"
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
            uid_metrics_key = f'id_metrics_{i}_metrics'
            
            if uid_metrics_key in problem and metrics_key in problem:
                outputs.append({
                    'index': i,
                    'uid_metrics': problem[uid_metrics_key],
                    'accuracy': problem[metrics_key].get('math_equal', 0),
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

def create_boxplots(summary, results, outdir, filename_prefix, input_filename=None, overall_metrics=None):
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

    # Add dotted lines for overall metrics if provided
    if overall_metrics:
        overall_accuracy = overall_metrics.get('overall_mean_accuracy', None)
        overall_self_certainty = overall_metrics.get('overall_mean_self_certainty_accuracy', None)
        overall_cot_decoding = overall_metrics.get('overall_mean_cot_decoding_accuracy', None)
        overall_confidence = overall_metrics.get('overall_mean_confidence_accuracy', None)
        overall_entropy = overall_metrics.get('overall_mean_entropy_accuracy', None)
        
        if overall_accuracy is not None:
            ax2.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, 
                       label=f'Overall Accuracy ({overall_accuracy:.3f})', alpha=0.8)
        
        if overall_self_certainty is not None:
            ax2.axhline(y=overall_self_certainty, color='green', linestyle='--', linewidth=2, 
                       label=f'Overall Self-Certainty ({overall_self_certainty:.3f})', alpha=0.8)

        if overall_cot_decoding is not None:
            ax2.axhline(y=overall_cot_decoding, color='blue', linestyle='--', linewidth=2, 
                       label=f'Overall Cot-Decoding ({overall_cot_decoding:.3f})', alpha=0.8)

        if overall_confidence is not None:
            ax2.axhline(y=overall_confidence, color='yellow', linestyle='--', linewidth=2, 
                       label=f'Overall Confidence ({overall_confidence:.3f})', alpha=0.8)

        if overall_entropy is not None:
            ax2.axhline(y=overall_entropy, color='purple', linestyle='--', linewidth=2, 
                       label=f'Overall Entropy ({overall_entropy:.3f})', alpha=0.8)

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


def create_correlation_plot(data, outdir, filename_prefix, input_filename=None):
    """Create correlation plots between UID metrics and answer correctness."""
    
    # Define UID metrics to analyze (focusing on _equal metrics as requested)
    uid_metrics = [
        "uid_variance_equal", "uid_gini_equal", "uid_shannon_equal"
    ]
    
    # Collect all data points
    all_data = []
    
    for problem in data:
        problem_id = problem.get('id', 'unknown')
        
        # Find all outputs for this problem
        i = 0
        while f'Output_{i}' in problem:
            output_key = f'Output_{i}'
            metrics_key = f'Metrics_{i}'
            uid_metrics_key = f'id_metrics_{i}_metrics'
            
            if uid_metrics_key in problem and metrics_key in problem:
                uid_data = problem[uid_metrics_key]
                metrics_data = problem[metrics_key]
                
                data_point = {
                    'problem_id': problem_id,
                    'output_index': i,
                    'accuracy': metrics_data.get('math_equal', 0),
                    'exact_match': metrics_data.get('em', 0),
                    'f1': metrics_data.get('f1', 0)
                }
                
                # Add UID metrics
                for metric in uid_metrics:
                    data_point[metric] = uid_data.get(metric, 0)
                
                all_data.append(data_point)
            i += 1
    
    if not all_data:
        print("No data found for correlation analysis")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    
    # Create correlation plots
    fig, axes = plt.subplots(1, 1, figsize=(16, 12))
    if input_filename:
        dataset_name = input_filename.split("/")[-2] if "/" in input_filename else input_filename.replace(".json", "")
        fig.suptitle('UID Metrics vs Answer Correctness Correlation \n ' + dataset_name, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('UID Metrics vs Answer Correctness Correlation \n ' + filename_prefix, fontsize=16, fontweight='bold')
    
    # Plot 4: Correlation heatmap
    ax4 = axes
    correlation_data = []
    metric_names = []
    
    for metric in uid_metrics:
        if SCIPY_AVAILABLE:
            acc_corr, _ = pearsonr(df[metric], df['accuracy'])
            em_corr, _ = pearsonr(df[metric], df['exact_match'])
            f1_corr, _ = pearsonr(df[metric], df['f1'])
        else:
            # Use numpy correlation as fallback
            acc_corr = np.corrcoef(df[metric], df['accuracy'])[0, 1]
            em_corr = np.corrcoef(df[metric], df['exact_match'])[0, 1]
            f1_corr = np.corrcoef(df[metric], df['f1'])[0, 1]
        
        correlation_data.append([acc_corr, em_corr, f1_corr])
        metric_names.append(metric.replace('uid_', '').replace('_equal', '').title())
    
    correlation_matrix = np.array(correlation_data)
    im = ax4.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    
    # Add text annotations
    for i in range(len(metric_names)):
        for j in range(3):
            text = ax4.text(j, i, f'{correlation_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontweight='bold', fontsize=32)
    
    ax4.set_xticks([0, 1, 2])
    ax4.set_xticklabels(['Accuracy', 'Exact Match', 'F1'])
    ax4.set_yticks(range(len(metric_names)))
    ax4.set_yticklabels(metric_names)
    ax4.set_title('Correlation Heatmap')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label('Correlation Coefficient')
    
    plt.tight_layout()
    
    # Save the plot
    plot_file = os.path.join(outdir, f"{filename_prefix}_uid_correlation.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print correlation statistics
    print("\n" + "="*80)
    print("UID METRICS CORRELATION WITH ANSWER CORRECTNESS")
    print("="*80)
    
    for metric in uid_metrics:
        if SCIPY_AVAILABLE:
            acc_corr, acc_p = pearsonr(df[metric], df['accuracy'])
            em_corr, em_p = pearsonr(df[metric], df['exact_match'])
            f1_corr, f1_p = pearsonr(df[metric], df['f1'])
        else:
            # Use numpy correlation as fallback (no p-values available)
            acc_corr = np.corrcoef(df[metric], df['accuracy'])[0, 1]
            em_corr = np.corrcoef(df[metric], df['exact_match'])[0, 1]
            f1_corr = np.corrcoef(df[metric], df['f1'])[0, 1]
            acc_p = em_p = f1_p = None
        
        print(f"\n{metric}:")
        if SCIPY_AVAILABLE:
            print(f"  vs Accuracy:     r = {acc_corr:.4f}, p = {acc_p:.4f}")
            print(f"  vs Exact Match:  r = {em_corr:.4f}, p = {em_p:.4f}")
            print(f"  vs F1 Score:     r = {f1_corr:.4f}, p = {f1_p:.4f}")
        else:
            print(f"  vs Accuracy:     r = {acc_corr:.4f}")
            print(f"  vs Exact Match:  r = {em_corr:.4f}")
            print(f"  vs F1 Score:     r = {f1_corr:.4f}")
    
    return plot_file


def extract_shannon_equal_outputs(data, outdir, filename_prefix):
    """Extract and save the single highest and lowest outputs for uid_shannon_equal metric across all problems."""
    
    all_outputs = []
    
    for problem in data:
        problem_id = problem.get('id', 'unknown')
        
        # Find all outputs for this problem
        i = 0
        while f'Output_{i}' in problem:
            output_key = f'Output_{i}'
            metrics_key = f'Metrics_{i}'
            uid_metrics_key = f'id_metrics_{i}_metrics'
            
            if uid_metrics_key in problem and metrics_key in problem:
                uid_data = problem[uid_metrics_key]
                metrics_data = problem[metrics_key]
                
                all_outputs.append({
                    'problem_id': problem_id,
                    'output_index': i,
                    'output': problem[output_key],
                    'uid_shannon_equal': uid_data.get('uid_shannon_equal', 0),
                    'accuracy': metrics_data.get('math_equal', 0),
                    'exact_match': metrics_data.get('em', 0),
                    'f1': metrics_data.get('f1', 0)
                })
            i += 1
    
    if not all_outputs:
        print("No data found for Shannon Equal analysis")
        return None
    
    # Find the single highest and lowest uid_shannon_equal outputs across all problems
    highest_output = max(all_outputs, key=lambda x: x['uid_shannon_equal'])
    lowest_output = min(all_outputs, key=lambda x: x['uid_shannon_equal'])
    
    shannon_outputs = {
        'highest_shannon_equal': {
            'problem_id': highest_output['problem_id'],
            'output_index': highest_output['output_index'],
            'uid_shannon_equal_score': highest_output['uid_shannon_equal'],
            'output_text': highest_output['output'],
            'accuracy': highest_output['accuracy'],
            'exact_match': highest_output['exact_match'],
            'f1': highest_output['f1']
        },
        'lowest_shannon_equal': {
            'problem_id': lowest_output['problem_id'],
            'output_index': lowest_output['output_index'],
            'uid_shannon_equal_score': lowest_output['uid_shannon_equal'],
            'output_text': lowest_output['output'],
            'accuracy': lowest_output['accuracy'],
            'exact_match': lowest_output['exact_match'],
            'f1': lowest_output['f1']
        }
    }
    
    # Save to JSON file
    shannon_file = os.path.join(outdir, f"{filename_prefix}_shannon_equal_outputs.json")
    with open(shannon_file, "w") as f:
        json.dump(shannon_outputs, f, indent=2)
    
    print(f"\nShannon Equal outputs saved to: {shannon_file}")
    print(f"Highest Shannon Equal: Problem {highest_output['problem_id']}, Output {highest_output['output_index']}, Score: {highest_output['uid_shannon_equal']:.4f}")
    print(f"Lowest Shannon Equal: Problem {lowest_output['problem_id']}, Output {lowest_output['output_index']}, Score: {lowest_output['uid_shannon_equal']:.4f}")
    
    return shannon_file


def analyze_uid_accuracy_by_level(data):
    """
    Analyze accuracy based on UID scores, grouped by difficulty level.
    For each problem, find outputs with highest and lowest UID scores for each metric,
    then calculate their accuracy, grouped by level.
    """
    
    # Define UID metrics to analyze
    uid_metrics = [
        "uid_variance_equal", "uid_gini_equal", "uid_shannon_equal",
        "uid_variance_logprob", "uid_gini_logprob", "uid_shannon_logprob", 
        "uid_variance_entropy", "uid_gini_entropy", "uid_shannon_entropy",
        "uid_variance_confidence_gap", "uid_gini_confidence_gap", "uid_shannon_confidence_gap",
        "uid_l2_equal", "uid_l2_logprob", "uid_l2_entropy", "uid_l2_confidence_gap"
    ]
    
    # Group problems by level
    problems_by_level = defaultdict(list)
    for problem in data:
        level = problem.get('level', 'unknown')
        problems_by_level[level].append(problem)
    
    level_results = {}
    
    for level, level_problems in problems_by_level.items():
        results = {
            'highest_uid': defaultdict(list),
            'lowest_uid': defaultdict(list)
        }
        
        for problem in level_problems:
            problem_id = problem.get('id', 'unknown')
            
            # Find all outputs for this problem
            outputs = []
            i = 0
            while f'Output_{i}' in problem:
                output_key = f'Output_{i}'
                metrics_key = f'Metrics_{i}'
                uid_metrics_key = f'id_metrics_{i}_metrics'
                
                if uid_metrics_key in problem and metrics_key in problem:
                    outputs.append({
                        'index': i,
                        'uid_metrics': problem[uid_metrics_key],
                        'accuracy': problem[metrics_key].get('math_equal', 0),
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
        
        level_results[level] = results
    
    return level_results


def create_level_boxplots(level_summary, level_results, outdir, filename_prefix, input_filename=None, overall_metrics=None, level_specific_metrics=None):
    """Create boxplots for each level showing UID analysis results."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get all levels
    levels = sorted(level_summary.keys())
    
    # Create figure with subplots for each level
    fig, axes = plt.subplots(len(levels), 1, figsize=(16, 6 * len(levels)))
    if len(levels) == 1:
        axes = [axes]
    
    if input_filename:
        dataset_name = input_filename.split("/")[-2] if "/" in input_filename else input_filename.replace(".json", "")
        fig.suptitle('UID Analysis by Difficulty Level: Mean Accuracy by Individual Metric \n ' + dataset_name, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('UID Analysis by Difficulty Level: Mean Accuracy by Individual Metric \n ' + filename_prefix, fontsize=16, fontweight='bold')
    
    plot_files = []  # Track individual plot files
    
    for i, level in enumerate(levels):
        ax = axes[i]
        summary = level_summary[level]
        
        # Get metrics for this level
        metrics = list(summary['highest_uid'].keys())
        metric_labels = [metric.replace('uid_', '').replace('_', ' ').title() for metric in metrics]
        
        # Get data for this level
        highest_mean_acc = [summary['highest_uid'][metric]['mean_accuracy'] for metric in metrics]
        lowest_mean_acc = [summary['lowest_uid'][metric]['mean_accuracy'] for metric in metrics]
        
        x = np.arange(len(metric_labels))
        width = 0.35
        
        bars_highest = ax.bar(x - width/2, highest_mean_acc, width, label='Highest UID', color='lightblue', alpha=0.8)
        bars_lowest = ax.bar(x + width/2, lowest_mean_acc, width, label='Lowest UID', color='lightcoral', alpha=0.8)

        # Add value labels on top of bars
        for bar, value in zip(bars_highest, highest_mean_acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        for bar, value in zip(bars_lowest, lowest_mean_acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Add dotted lines for level-specific metrics if provided
        if level_specific_metrics and str(level) in level_specific_metrics:
            level_metrics = level_specific_metrics[str(level)]
            level_accuracy = level_metrics.get('math_equal', None)
            level_self_certainty = level_metrics.get('self_certainty_accuracy', None)
            
            if level_accuracy is not None:
                ax.axhline(y=level_accuracy, color='red', linestyle='--', linewidth=2, 
                           label=f'Level {level} Accuracy ({level_accuracy:.3f})', alpha=0.8)
            
            if level_self_certainty is not None:
                ax.axhline(y=level_self_certainty, color='green', linestyle='--', linewidth=2, 
                           label=f'Level {level} Self-Certainty ({level_self_certainty:.3f})', alpha=0.8)
        # Fallback to overall metrics if level-specific not available
        elif overall_metrics:
            overall_accuracy = overall_metrics.get('overall_mean_accuracy', None)
            overall_self_certainty = overall_metrics.get('overall_mean_self_certainty_accuracy', None)
            overall_cot_decoding = overall_metrics.get('overall_mean_cot_decoding_accuracy', None)
            overall_confidence = overall_metrics.get('overall_mean_confidence_accuracy', None)
            overall_entropy = overall_metrics.get('overall_mean_entropy_accuracy', None)
            
            if overall_accuracy is not None:
                ax.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, 
                           label=f'Overall Accuracy ({overall_accuracy:.3f})', alpha=0.8)
            
            if overall_self_certainty is not None:
                ax.axhline(y=overall_self_certainty, color='green', linestyle='--', linewidth=2, 
                           label=f'Overall Self-Certainty ({overall_self_certainty:.3f})', alpha=0.8)

            if overall_cot_decoding is not None:
                ax.axhline(y=overall_cot_decoding, color='blue', linestyle='--', linewidth=2, 
                           label=f'Overall Cot-Decoding ({overall_cot_decoding:.3f})', alpha=0.8)

            if overall_confidence is not None:
                ax.axhline(y=overall_confidence, color='yellow', linestyle='--', linewidth=2, 
                           label=f'Overall Confidence ({overall_confidence:.3f})', alpha=0.8)

            if overall_entropy is not None:
                ax.axhline(y=overall_entropy, color='purple', linestyle='--', linewidth=2, 
                           label=f'Overall Entropy ({overall_entropy:.3f})', alpha=0.8)

        ax.set_xlabel('UID Metrics')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title(f'Level {level} - Mean Accuracy by UID Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Create individual plot for this level
        plt.figure(figsize=(12, 8))
        individual_ax = plt.gca()
        
        bars_highest_ind = individual_ax.bar(x - width/2, highest_mean_acc, width, label='Highest UID', color='lightblue', alpha=0.8)
        bars_lowest_ind = individual_ax.bar(x + width/2, lowest_mean_acc, width, label='Lowest UID', color='lightcoral', alpha=0.8)

        # Add value labels on top of bars for individual plot
        for bar, value in zip(bars_highest_ind, highest_mean_acc):
            individual_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        for bar, value in zip(bars_lowest_ind, lowest_mean_acc):
            individual_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Add dotted lines for level-specific metrics if provided (for individual plots too)
        if level_specific_metrics and str(level) in level_specific_metrics:
            level_metrics = level_specific_metrics[str(level)]
            level_accuracy = level_metrics.get('math_equal', None)
            level_self_certainty = level_metrics.get('self_certainty_accuracy', None)
            level_cot_decoding = level_metrics.get('cot_decoding_accuracy', None)
            level_confidence = level_metrics.get('confidence_accuracy', None)
            level_entropy = level_metrics.get('entropy_accuracy', None)
            
            if level_accuracy is not None:
                individual_ax.axhline(y=level_accuracy, color='red', linestyle='--', linewidth=2, 
                                     label=f'Level {level} Accuracy ({level_accuracy:.3f})', alpha=0.8)
            
            if level_self_certainty is not None:
                individual_ax.axhline(y=level_self_certainty, color='green', linestyle='--', linewidth=2, 
                                     label=f'Level {level} Self-Certainty ({level_self_certainty:.3f})', alpha=0.8)
            
            if level_cot_decoding is not None:
                individual_ax.axhline(y=level_cot_decoding, color='blue', linestyle='--', linewidth=2, 
                                     label=f'Level {level} Cot-Decoding ({level_cot_decoding:.3f})', alpha=0.8)
            
            if level_confidence is not None:
                individual_ax.axhline(y=level_confidence, color='yellow', linestyle='--', linewidth=2, 
                                     label=f'Level {level} Confidence ({level_confidence:.3f})', alpha=0.8)
            
            if level_entropy is not None:
                individual_ax.axhline(y=level_entropy, color='purple', linestyle='--', linewidth=2, 
                                     label=f'Level {level} Entropy ({level_entropy:.3f})', alpha=0.8)
        # Fallback to overall metrics if level-specific not available
        elif overall_metrics:
            overall_accuracy = overall_metrics.get('overall_mean_accuracy', None)
            overall_self_certainty = overall_metrics.get('overall_mean_self_certainty_accuracy', None)
            overall_cot_decoding = overall_metrics.get('overall_mean_cot_decoding_accuracy', None)
            overall_confidence = overall_metrics.get('overall_mean_confidence_accuracy', None)
            overall_entropy = overall_metrics.get('overall_mean_entropy_accuracy', None)
            
            if overall_accuracy is not None:
                individual_ax.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, 
                                     label=f'Overall Accuracy ({overall_accuracy:.3f})', alpha=0.8)
            
            if overall_self_certainty is not None:
                individual_ax.axhline(y=overall_self_certainty, color='green', linestyle='--', linewidth=2, 
                                     label=f'Overall Self-Certainty ({overall_self_certainty:.3f})', alpha=0.8)

            if overall_cot_decoding is not None:
                individual_ax.axhline(y=overall_cot_decoding, color='blue', linestyle='--', linewidth=2, 
                                     label=f'Overall Cot-Decoding ({overall_cot_decoding:.3f})', alpha=0.8)

            if overall_confidence is not None:
                individual_ax.axhline(y=overall_confidence, color='yellow', linestyle='--', linewidth=2, 
                                     label=f'Overall Confidence ({overall_confidence:.3f})', alpha=0.8)

            if overall_entropy is not None:
                individual_ax.axhline(y=overall_entropy, color='purple', linestyle='--', linewidth=2, 
                                     label=f'Overall Entropy ({overall_entropy:.3f})', alpha=0.8)

        individual_ax.set_xlabel('UID Metrics')
        individual_ax.set_ylabel('Mean Accuracy')
        individual_ax.set_title(f'Level {level} - UID Analysis: Mean Accuracy by Individual Metric')
        individual_ax.set_xticks(x)
        individual_ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        individual_ax.legend()
        individual_ax.grid(True, alpha=0.3)
        individual_ax.set_ylim(0, 1.1)
        
        # Save individual level plot
        individual_plot_file = os.path.join(outdir, f"{filename_prefix}_uid_analysis_level_{level}.png")
        plt.savefig(individual_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(individual_plot_file)
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_file = os.path.join(outdir, f"{filename_prefix}_uid_analysis_by_level.png")
    plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_plot_file, plot_files


def analyze_uid_accuracy_by_domain(data):
    """
    Analyze accuracy based on UID scores, grouped by high-level domain.
    For each problem, find outputs with highest and lowest UID scores for each metric,
    then calculate their accuracy, grouped by domain.
    """
    
    # Define UID metrics to analyze
    uid_metrics = [
        "uid_variance_equal", "uid_gini_equal", "uid_shannon_equal",
        "uid_variance_logprob", "uid_gini_logprob", "uid_shannon_logprob", 
        "uid_variance_entropy", "uid_gini_entropy", "uid_shannon_entropy",
        "uid_variance_confidence_gap", "uid_gini_confidence_gap", "uid_shannon_confidence_gap",
        "uid_l2_equal", "uid_l2_logprob", "uid_l2_entropy", "uid_l2_confidence_gap"
    ]
    
    # Group problems by domain
    problems_by_domain = defaultdict(list)
    for problem in data:
        domain = problem.get('High-level domain', 'unknown')
        problems_by_domain[domain].append(problem)
    
    domain_results = {}
    
    for domain, domain_problems in problems_by_domain.items():
        results = {
            'highest_uid': defaultdict(list),
            'lowest_uid': defaultdict(list)
        }
        
        for problem in domain_problems:
            problem_id = problem.get('id', 'unknown')
            
            # Find all outputs for this problem
            outputs = []
            i = 0
            while f'Output_{i}' in problem:
                output_key = f'Output_{i}'
                metrics_key = f'Metrics_{i}'
                uid_metrics_key = f'id_metrics_{i}_metrics'
                
                if uid_metrics_key in problem and metrics_key in problem:
                    outputs.append({
                        'index': i,
                        'uid_metrics': problem[uid_metrics_key],
                        'accuracy': problem[metrics_key].get('math_equal', 0),
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
        
        domain_results[domain] = results
    
    return domain_results


def create_domain_boxplots(domain_summary, domain_results, outdir, filename_prefix, input_filename=None, overall_metrics=None):
    """Create boxplots for each domain showing UID analysis results."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Get all domains
    domains = sorted(domain_summary.keys())
    
    # Create figure with subplots for each domain
    fig, axes = plt.subplots(len(domains), 1, figsize=(16, 6 * len(domains)))
    if len(domains) == 1:
        axes = [axes]
    
    if input_filename:
        dataset_name = input_filename.split("/")[-2] if "/" in input_filename else input_filename.replace(".json", "")
        fig.suptitle('UID Analysis by Domain: Mean Accuracy by Individual Metric \n ' + dataset_name, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('UID Analysis by Domain: Mean Accuracy by Individual Metric \n ' + filename_prefix, fontsize=16, fontweight='bold')
    
    plot_files = []  # Track individual plot files
    
    for i, domain in enumerate(domains):
        ax = axes[i]
        summary = domain_summary[domain]
        
        # Get metrics for this domain
        metrics = list(summary['highest_uid'].keys())
        metric_labels = [metric.replace('uid_', '').replace('_', ' ').title() for metric in metrics]
        
        # Get data for this domain
        highest_mean_acc = [summary['highest_uid'][metric]['mean_accuracy'] for metric in metrics]
        lowest_mean_acc = [summary['lowest_uid'][metric]['mean_accuracy'] for metric in metrics]
        
        x = np.arange(len(metric_labels))
        width = 0.35
        
        bars_highest = ax.bar(x - width/2, highest_mean_acc, width, label='Highest UID', color='lightblue', alpha=0.8)
        bars_lowest = ax.bar(x + width/2, lowest_mean_acc, width, label='Lowest UID', color='lightcoral', alpha=0.8)

        # Add value labels on top of bars
        for bar, value in zip(bars_highest, highest_mean_acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        for bar, value in zip(bars_lowest, lowest_mean_acc):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Add dotted lines for overall metrics if provided
        if overall_metrics:
            overall_accuracy = overall_metrics.get('overall_mean_accuracy', None)
            overall_self_certainty = overall_metrics.get('overall_mean_self_certainty_accuracy', None)
            overall_cot_decoding = overall_metrics.get('overall_mean_cot_decoding_accuracy', None)
            overall_confidence = overall_metrics.get('overall_mean_confidence_accuracy', None)
            overall_entropy = overall_metrics.get('overall_mean_entropy_accuracy', None)
            
            if overall_accuracy is not None:
                ax.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, 
                           label=f'Overall Accuracy ({overall_accuracy:.3f})', alpha=0.8)
            
            if overall_self_certainty is not None:
                ax.axhline(y=overall_self_certainty, color='green', linestyle='--', linewidth=2, 
                           label=f'Overall Self-Certainty ({overall_self_certainty:.3f})', alpha=0.8)

            if overall_cot_decoding is not None:
                ax.axhline(y=overall_cot_decoding, color='blue', linestyle='--', linewidth=2, 
                           label=f'Overall Cot-Decoding ({overall_cot_decoding:.3f})', alpha=0.8)

            if overall_confidence is not None:
                ax.axhline(y=overall_confidence, color='yellow', linestyle='--', linewidth=2, 
                           label=f'Overall Confidence ({overall_confidence:.3f})', alpha=0.8)

            if overall_entropy is not None:
                ax.axhline(y=overall_entropy, color='purple', linestyle='--', linewidth=2, 
                           label=f'Overall Entropy ({overall_entropy:.3f})', alpha=0.8)

        ax.set_xlabel('UID Metrics')
        ax.set_ylabel('Mean Accuracy')
        ax.set_title(f'Domain {domain} - Mean Accuracy by UID Metric')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
        
        # Create individual plot for this domain
        plt.figure(figsize=(12, 8))
        individual_ax = plt.gca()
        
        bars_highest_ind = individual_ax.bar(x - width/2, highest_mean_acc, width, label='Highest UID', color='lightblue', alpha=0.8)
        bars_lowest_ind = individual_ax.bar(x + width/2, lowest_mean_acc, width, label='Lowest UID', color='lightcoral', alpha=0.8)

        # Add value labels on top of bars for individual plot
        for bar, value in zip(bars_highest_ind, highest_mean_acc):
            individual_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        for bar, value in zip(bars_lowest_ind, lowest_mean_acc):
            individual_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                              f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        # Add dotted lines for overall metrics if provided (for individual plots too)
        if overall_metrics:
            overall_accuracy = overall_metrics.get('overall_mean_accuracy', None)
            overall_self_certainty = overall_metrics.get('overall_mean_self_certainty_accuracy', None)
            overall_cot_decoding = overall_metrics.get('overall_mean_cot_decoding_accuracy', None)
            overall_confidence = overall_metrics.get('overall_mean_confidence_accuracy', None)
            overall_entropy = overall_metrics.get('overall_mean_entropy_accuracy', None)
            
            if overall_accuracy is not None:
                individual_ax.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, 
                                     label=f'Overall Accuracy ({overall_accuracy:.3f})', alpha=0.8)
            
            if overall_self_certainty is not None:
                individual_ax.axhline(y=overall_self_certainty, color='green', linestyle='--', linewidth=2, 
                                     label=f'Overall Self-Certainty ({overall_self_certainty:.3f})', alpha=0.8)

            if overall_cot_decoding is not None:
                individual_ax.axhline(y=overall_cot_decoding, color='blue', linestyle='--', linewidth=2, 
                                     label=f'Overall Cot-Decoding ({overall_cot_decoding:.3f})', alpha=0.8)

            if overall_confidence is not None:
                individual_ax.axhline(y=overall_confidence, color='yellow', linestyle='--', linewidth=2, 
                                     label=f'Overall Confidence ({overall_confidence:.3f})', alpha=0.8)

            if overall_entropy is not None:
                individual_ax.axhline(y=overall_entropy, color='purple', linestyle='--', linewidth=2, 
                                     label=f'Overall Entropy ({overall_entropy:.3f})', alpha=0.8)

        individual_ax.set_xlabel('UID Metrics')
        individual_ax.set_ylabel('Mean Accuracy')
        individual_ax.set_title(f'Domain {domain} - UID Analysis: Mean Accuracy by Individual Metric')
        individual_ax.set_xticks(x)
        individual_ax.set_xticklabels(metric_labels, rotation=45, ha='right')
        individual_ax.legend()
        individual_ax.grid(True, alpha=0.3)
        individual_ax.set_ylim(0, 1.1)
        
        # Save individual domain plot
        individual_plot_file = os.path.join(outdir, f"{filename_prefix}_uid_analysis_domain_{domain}.png")
        plt.savefig(individual_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(individual_plot_file)
    
    plt.tight_layout()
    
    # Save the combined plot
    combined_plot_file = os.path.join(outdir, f"{filename_prefix}_uid_analysis_by_domain.png")
    plt.savefig(combined_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return combined_plot_file, plot_files


def collect_all_differences(data):
    """Collect all differences from all traces to calculate global thresholds"""
    all_positive_diffs = []
    all_negative_diffs = []
    
    total_id_data_found = 0
    
    for item in data:
        i = 0
        while True:
            h_key = f"id_h_{i}"
            if h_key not in item:
                break
                
            id_h = item.get(h_key, [])
            if len(id_h) >= 2:
                total_id_data_found += 1
                prev = id_h[0]
                for x in id_h[1:]:
                    try:
                        dx = float(x) - float(prev)
                        if dx > 0:
                            all_positive_diffs.append(dx)
                        elif dx < 0:
                            all_negative_diffs.append(dx)
                    except Exception:
                        pass
                    prev = x
            i += 1
    
    print(f"Found {total_id_data_found} id_h arrays with sufficient data")
    print(f"Collected {len(all_positive_diffs)} positive differences and {len(all_negative_diffs)} negative differences")
    
    avg_positive = sum(all_positive_diffs) / len(all_positive_diffs) if all_positive_diffs else 0
    avg_negative = sum(all_negative_diffs) / len(all_negative_diffs) if all_negative_diffs else 0
    
    return avg_positive, avg_negative


def calculate_sigma_thresholds(data):
    """Calculate proper sigma-based thresholds for spike/fall detection"""
    all_diffs = []
    
    for item in data:
        i = 0
        while True:
            h_key = f"id_h_{i}"
            if h_key not in item:
                break
                
            id_h = item.get(h_key, [])
            if len(id_h) >= 2:
                prev = id_h[0]
                for x in id_h[1:]:
                    try:
                        dx = float(x) - float(prev)
                        all_diffs.append(dx)
                    except Exception:
                        pass
                    prev = x
            i += 1
    
    print(f"calculate_sigma_thresholds: Collected {len(all_diffs)} total differences")
    
    if not all_diffs:
        print("calculate_sigma_thresholds: No differences found, returning zeros")
        return 0, 0, 0, 0, 0, 0
    
    # Calculate mean and standard deviation
    mean_diff = sum(all_diffs) / len(all_diffs)
    variance = sum((x - mean_diff) ** 2 for x in all_diffs) / len(all_diffs)
    std_diff = variance ** 0.5
    
    print(f"calculate_sigma_thresholds: Mean={mean_diff:.6f}, Std={std_diff:.6f}")
    
    # Calculate thresholds
    avg_positive, avg_negative = collect_all_differences(data)
    
    # For sigma-based thresholds, use mean ± n*std
    threshold_2sigma_pos = mean_diff + 2 * std_diff
    threshold_2sigma_neg = mean_diff - 2 * std_diff
    threshold_3sigma_pos = mean_diff + 3 * std_diff
    threshold_3sigma_neg = mean_diff - 3 * std_diff
    
    print(f"calculate_sigma_thresholds: 2-sigma thresholds: pos={threshold_2sigma_pos:.6f}, neg={threshold_2sigma_neg:.6f}")
    print(f"calculate_sigma_thresholds: 3-sigma thresholds: pos={threshold_3sigma_pos:.6f}, neg={threshold_3sigma_neg:.6f}")
    
    return avg_positive, avg_negative, threshold_2sigma_pos, threshold_2sigma_neg, threshold_3sigma_pos, threshold_3sigma_neg


def count_spikes_falls(arr, avg_positive, avg_negative):
    """Count spikes and falls based on average thresholds"""
    if not isinstance(arr, list) or len(arr) < 2:
        return {"spikes": 0, "falls": 0, "zeros": 0}
    
    spikes = falls = zeros = 0
    prev = arr[0]
    
    for x in arr[1:]:
        try:
            dx = float(x) - float(prev)
            if dx > avg_positive:
                spikes += 1
            elif dx < avg_negative:
                falls += 1
            else:
                zeros += 1
        except Exception:
            pass
        prev = x
    
    return {"spikes": spikes, "falls": falls, "zeros": zeros}


def analyze_spikes_falls_accuracy(data, avg_positive, avg_negative):
    """
    Analyze accuracy based on combined spike and fall counts.
    For each problem, find outputs with highest and lowest combined spike+fall counts,
    then calculate their accuracy.
    """
    
    results = {
        'highest_spikes_falls': defaultdict(list),
        'lowest_spikes_falls': defaultdict(list)
    }
    
    problems_with_data = 0
    total_problems = len(data)
    
    for problem in data:
        problem_id = problem.get('id', 'unknown')
        
        # Find all outputs for this problem
        outputs = []
        i = 0
        while f'Output_{i}' in problem:
            output_key = f'Output_{i}'
            metrics_key = f'Metrics_{i}'
            id_h_key = f'id_h_{i}'
            
            if id_h_key in problem and metrics_key in problem:
                id_h = problem[id_h_key]
                metrics_data = problem[metrics_key]
                
                # Only process if id_h has valid values (not empty and has at least 2 elements)
                if id_h and len(id_h) >= 2:
                    # Calculate spikes and falls for this output
                    spike_fall_counts = count_spikes_falls(id_h, avg_positive, avg_negative)
                    
                    outputs.append({
                        'index': i,
                        'spikes': spike_fall_counts['spikes'],
                        'falls': spike_fall_counts['falls'],
                        'zeros': spike_fall_counts['zeros'],
                        'combined_count': spike_fall_counts['spikes'] + spike_fall_counts['falls'],
                        'accuracy': metrics_data.get('math_equal', 0),
                        'exact_match': metrics_data.get('em', 0),
                        'f1': metrics_data.get('f1', 0)
                    })
                    problems_with_data += 1
            i += 1
        
        if not outputs:
            continue
            
        # Find output with highest combined spikes + falls
        highest_combined_output = max(outputs, key=lambda x: x['combined_count'])
        results['highest_spikes_falls']['combined'].append({
            'problem_id': problem_id,
            'output_index': highest_combined_output['index'],
            'combined_count': highest_combined_output['combined_count'],
            'spikes_count': highest_combined_output['spikes'],
            'falls_count': highest_combined_output['falls'],
            'accuracy': highest_combined_output['accuracy'],
            'exact_match': highest_combined_output['exact_match'],
            'f1': highest_combined_output['f1']
        })
        
        # Find output with lowest combined spikes + falls
        lowest_combined_output = min(outputs, key=lambda x: x['combined_count'])
        results['lowest_spikes_falls']['combined'].append({
            'problem_id': problem_id,
            'output_index': lowest_combined_output['index'],
            'combined_count': lowest_combined_output['combined_count'],
            'spikes_count': lowest_combined_output['spikes'],
            'falls_count': lowest_combined_output['falls'],
            'accuracy': lowest_combined_output['accuracy'],
            'exact_match': lowest_combined_output['exact_match'],
            'f1': lowest_combined_output['f1']
        })
    
    print(f"Found id_h data in {problems_with_data} outputs out of {total_problems} problems")
    print(f"Spike/fall analysis results: {len(results['highest_spikes_falls']['combined'])} highest, {len(results['lowest_spikes_falls']['combined'])} lowest")
    return results


def calculate_spikes_falls_summary_stats(results):
    """Calculate summary statistics for combined spike/fall results."""
    summary = {}
    
    for analysis_type in ['highest_spikes_falls', 'lowest_spikes_falls']:
        summary[analysis_type] = {}
        
        for metric in results[analysis_type]:
            accuracies = [item['accuracy'] for item in results[analysis_type][metric]]
            exact_matches = [item['exact_match'] for item in results[analysis_type][metric]]
            f1_scores = [item['f1'] for item in results[analysis_type][metric]]
            combined_counts = [item['combined_count'] for item in results[analysis_type][metric]]
            spikes_counts = [item['spikes_count'] for item in results[analysis_type][metric]]
            falls_counts = [item['falls_count'] for item in results[analysis_type][metric]]
            
            # Convert boolean accuracies to float for proper statistics
            accuracy_floats = [float(acc) for acc in accuracies]
            
            summary[analysis_type][metric] = {
                'count': len(accuracies),
                'mean_accuracy': np.mean(accuracy_floats) if accuracy_floats else 0,
                'std_accuracy': np.std(accuracy_floats) if accuracy_floats else 0,
                'mean_exact_match': np.mean(exact_matches) if exact_matches else 0,
                'std_exact_match': np.std(exact_matches) if exact_matches else 0,
                'mean_f1': np.mean(f1_scores) if f1_scores else 0,
                'std_f1': np.std(f1_scores) if f1_scores else 0,
                'mean_combined_count': np.mean(combined_counts) if combined_counts else 0,
                'std_combined_count': np.std(combined_counts) if combined_counts else 0,
                'mean_spikes_count': np.mean(spikes_counts) if spikes_counts else 0,
                'std_spikes_count': np.std(spikes_counts) if spikes_counts else 0,
                'mean_falls_count': np.mean(falls_counts) if falls_counts else 0,
                'std_falls_count': np.std(falls_counts) if falls_counts else 0
            }
    
    return summary


def create_spikes_falls_boxplots(spikes_falls_summary, outdir, filename_prefix, input_filename=None, overall_metrics=None):
    """Create bar graphs for mean accuracy based on spike/fall analysis."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    if input_filename:
        dataset_name = input_filename.split("/")[-2] if "/" in input_filename else input_filename.replace(".json", "")
        fig.suptitle('Spike/Fall Analysis: Mean Accuracy by Analysis Type \n ' + dataset_name, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Spike/Fall Analysis: Mean Accuracy by Analysis Type \n ' + filename_prefix, fontsize=16, fontweight='bold')
    
    # Prepare data - now using the combined structure
    analysis_types = ['highest_spikes_falls', 'lowest_spikes_falls']
    type_labels = ['Highest Spikes+Falls', 'Lowest Spikes+Falls']
    
    mean_accuracies = []
    for analysis_type in analysis_types:
        if analysis_type in spikes_falls_summary:
            for metric in spikes_falls_summary[analysis_type]:
                mean_accuracies.append(spikes_falls_summary[analysis_type][metric]['mean_accuracy'])
                break  # Only need one metric per analysis type
    
    # Handle case where no spike/fall data is available
    if not mean_accuracies:
        ax.text(0.5, 0.5, 'No spike/fall data available\n(id_h data not found)', 
                ha='center', va='center', transform=ax.transAxes, 
                fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax.set_title('Spike/Fall Analysis: No Data Available')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save the plot
        plot_file = os.path.join(outdir, f"{filename_prefix}_spikes_falls_analysis.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_file
    
    x = np.arange(len(mean_accuracies))
    width = 0.6
    
    bars = ax.bar(x, mean_accuracies, width, color=['lightblue', 'lightcoral'], alpha=0.8)

    # Add value labels on top of bars
    for bar, value in zip(bars, mean_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    # Add dotted lines for overall metrics if provided
    if overall_metrics:
        overall_accuracy = overall_metrics.get('overall_mean_accuracy', None)
        
        if overall_accuracy is not None:
            ax.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, 
                       label=f'Overall Accuracy ({overall_accuracy:.3f})', alpha=0.8)

    ax.set_title('Mean Accuracy by Spike/Fall Analysis Type')
    ax.set_ylabel('Mean Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels[:len(mean_accuracies)], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Save the plot
    plot_file = os.path.join(outdir, f"{filename_prefix}_spikes_falls_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_combined_analysis_plot(summary, spikes_falls_summary, spikes_falls_summary_2sigma, spikes_falls_summary_3sigma, outdir, filename_prefix, input_filename=None, overall_metrics=None):
    """Create combined plot showing UID analysis and all three spike/fall threshold methods."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with 4 subplots: 1 for UID, 3 for spike/fall methods
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(24, 12))
    if input_filename:
        dataset_name = input_filename.split("/")[-1].replace(".json", "")
        fig.suptitle(f'UID and Spike/Fall Analysis (All Threshold Methods): {dataset_name}', fontsize=16, fontweight='bold')
    else:
        fig.suptitle('UID and Spike/Fall Analysis (All Threshold Methods)', fontsize=16, fontweight='bold')
    
    # Top-left plot: UID Analysis - Mean Accuracy by Metric
    uid_metrics = list(summary['highest_uid'].keys())
    metric_labels = [metric.replace('uid_', '').replace('_', ' ').title() for metric in uid_metrics]
    
    highest_mean_acc = [summary['highest_uid'][metric]['mean_accuracy'] for metric in uid_metrics]
    lowest_mean_acc = [summary['lowest_uid'][metric]['mean_accuracy'] for metric in uid_metrics]
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    bars_highest = ax1.bar(x - width/2, highest_mean_acc, width, label='Highest UID', color='lightblue', alpha=0.8)
    bars_lowest = ax1.bar(x + width/2, lowest_mean_acc, width, label='Lowest UID', color='lightcoral', alpha=0.8)

    # Add value labels on top of bars
    for bar, value in zip(bars_highest, highest_mean_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    for bar, value in zip(bars_lowest, lowest_mean_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

    # Add overall metrics as horizontal lines if available
    if overall_metrics:
        ax1.axhline(y=overall_metrics.get('overall_mean_accuracy', 0), color='red', linestyle='--', alpha=0.7, 
                   label=f'Overall Accuracy ({overall_metrics.get("overall_mean_accuracy", 0):.4f})')
        ax1.axhline(y=overall_metrics.get('overall_mean_self_certainty_accuracy', 0), color='blue', linestyle='--', alpha=0.7, 
                   label=f'Self-Certainty Accuracy ({overall_metrics.get("overall_mean_self_certainty_accuracy", 0):.4f})')
        ax1.axhline(y=overall_metrics.get('overall_mean_cot_decoding_accuracy', 0), color='green', linestyle='--', alpha=0.7, 
                   label=f'Cot-Decoding Accuracy ({overall_metrics.get("overall_mean_cot_decoding_accuracy", 0):.4f})')
        ax1.axhline(y=overall_metrics.get('overall_mean_confidence_accuracy', 0), color='yellow', linestyle='--', alpha=0.7, 
                   label=f'Confidence Accuracy ({overall_metrics.get("overall_mean_confidence_accuracy", 0):.4f})')
        ax1.axhline(y=overall_metrics.get('overall_mean_entropy_accuracy', 0), color='purple', linestyle='--', alpha=0.7, 
                   label=f'Entropy Accuracy ({overall_metrics.get("overall_mean_entropy_accuracy", 0):.4f})')
        ax1.axhline(y=overall_metrics.get('overall_mean_upper_bound_accuracy', 0), color='orange', linestyle='--', alpha=0.7, 
                   label=f'Upper Bound Accuracy ({overall_metrics.get("overall_mean_upper_bound_accuracy", 0):.4f})')
        ax1.legend()
    
    ax1.set_title('UID-based Accuracy Analysis', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Accuracy', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Helper function to create spike/fall plots - UPDATED for combined structure
    def create_spike_fall_plot(ax, spikes_falls_summary, title, overall_metrics):
        analysis_types = ['highest_spikes_falls', 'lowest_spikes_falls']
        type_labels = ['Highest Spikes+Falls', 'Lowest Spikes+Falls']
        
        mean_accuracies = []
        for analysis_type in analysis_types:
            if analysis_type in spikes_falls_summary:
                for metric in spikes_falls_summary[analysis_type]:
                    mean_accuracies.append(spikes_falls_summary[analysis_type][metric]['mean_accuracy'])
                    break  # Only need one metric per analysis type
        
        # Handle case where no spike/fall data is available
        if not mean_accuracies:
            ax.text(0.5, 0.5, 'No spike/fall data available\n(id_h data not found)', 
                    ha='center', va='center', transform=ax.transAxes, 
                    fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title(title + ' - No Data Available', fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])
            return
        
        x = np.arange(len(mean_accuracies))
        width = 0.6
        
        bars = ax.bar(x, mean_accuracies, width, color=['lightblue', 'lightcoral'], alpha=0.8)

        # Add value labels on top of bars
        for bar, value in zip(bars, mean_accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=8)

        # Add overall metrics as horizontal lines if available
        if overall_metrics:
            ax.axhline(y=overall_metrics.get('overall_mean_accuracy', 0), color='red', linestyle='--', alpha=0.7, 
                      label=f'Overall Accuracy ({overall_metrics.get("overall_mean_accuracy", 0):.4f})')
            ax.axhline(y=overall_metrics.get('overall_mean_self_certainty_accuracy', 0), color='blue', linestyle='--', alpha=0.7, 
                      label=f'Self-Certainty Accuracy ({overall_metrics.get("overall_mean_self_certainty_accuracy", 0):.4f})')
            ax.axhline(y=overall_metrics.get('overall_mean_cot_decoding_accuracy', 0), color='green', linestyle='--', alpha=0.7, 
                      label=f'Cot-Decoding Accuracy ({overall_metrics.get("overall_mean_cot_decoding_accuracy", 0):.4f})')
            ax.axhline(y=overall_metrics.get('overall_mean_upper_bound_accuracy', 0), color='orange', linestyle='--', alpha=0.7, 
                      label=f'Upper Bound Accuracy ({overall_metrics.get("overall_mean_upper_bound_accuracy", 0):.4f})')
            ax.axhline(y=overall_metrics.get('overall_mean_confidence_accuracy', 0), color='yellow', linestyle='--', alpha=0.7, 
                      label=f'Confidence Accuracy ({overall_metrics.get("overall_mean_confidence_accuracy", 0):.4f})')
            ax.axhline(y=overall_metrics.get('overall_mean_entropy_accuracy', 0), color='purple', linestyle='--', alpha=0.7, 
                      label=f'Entropy Accuracy ({overall_metrics.get("overall_mean_entropy_accuracy", 0):.4f})')
            ax.legend()
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Accuracy', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(type_labels[:len(mean_accuracies)], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    # Top-right plot: Average Thresholds
    create_spike_fall_plot(ax2, spikes_falls_summary, 'Spike/Fall Analysis (Average Thresholds)', overall_metrics)
    
    # Bottom-left plot: 2-Sigma Thresholds
    create_spike_fall_plot(ax3, spikes_falls_summary_2sigma, 'Spike/Fall Analysis (2-Sigma Thresholds)', overall_metrics)
    
    # Bottom-right plot: 3-Sigma Thresholds
    create_spike_fall_plot(ax4, spikes_falls_summary_3sigma, 'Spike/Fall Analysis (3-Sigma Thresholds)', overall_metrics)
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_file = os.path.join(outdir, f"{filename_prefix}_combined_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def filter_traces_by_criteria(data, avg_positive=None, avg_negative=None):
    """
    Filter out traces based on specific criteria for each question:
    1) Highest UID variance
    2) Lowest UID variance  
    3) Highest number of spikes and falls
    4) Lowest number of spikes and falls
    
    Returns filtered data with the specified traces removed.
    """
    filtered_data = []
    
    for problem in data:
        problem_id = problem.get('id', 'unknown')
        
        # Find all outputs for this problem
        outputs = []
        i = 0
        while f'Output_{i}' in problem:
            output_key = f'Output_{i}'
            metrics_key = f'Metrics_{i}'
            uid_metrics_key = f'id_metrics_{i}_metrics'
            id_h_key = f'id_h_{i}'
            
            output_data = {
                'index': i,
                'output_key': output_key,
                'metrics_key': metrics_key,
                'uid_metrics_key': uid_metrics_key,
                'id_h_key': id_h_key
            }
            
            # Get UID metrics if available
            if uid_metrics_key in problem:
                output_data['uid_metrics'] = problem[uid_metrics_key]
            
            # Get spike/fall data if available
            if id_h_key in problem and avg_positive is not None and avg_negative is not None:
                id_h = problem[id_h_key]
                spike_fall_counts = count_spikes_falls(id_h, avg_positive, avg_negative)
                output_data['spikes'] = spike_fall_counts['spikes']
                output_data['falls'] = spike_fall_counts['falls']
                output_data['combined_count'] = spike_fall_counts['spikes'] + spike_fall_counts['falls']
            
            outputs.append(output_data)
            i += 1
        
        if len(outputs) <= 1:
            # If only one or no outputs, keep the problem as is
            filtered_data.append(problem)
            continue
        
        # Create a copy of the problem
        filtered_problem = problem.copy()
        
        # Find traces to filter out
        traces_to_remove = set()
        
        # 1) Filter out trace with highest UID variance
        if any('uid_metrics' in output for output in outputs):
            uid_variance_values = []
            for output in outputs:
                if 'uid_metrics' in output:
                    uid_variance = output['uid_metrics'].get('uid_variance_equal', 0)
                    uid_variance_values.append((output['index'], uid_variance))
            
            if uid_variance_values:
                highest_variance_trace = max(uid_variance_values, key=lambda x: x[1])
                traces_to_remove.add(highest_variance_trace[0])
        
        # 2) Filter out trace with lowest UID variance
        if any('uid_metrics' in output for output in outputs):
            uid_variance_values = []
            for output in outputs:
                if 'uid_metrics' in output:
                    uid_variance = output['uid_metrics'].get('uid_variance_equal', 0)
                    uid_variance_values.append((output['index'], uid_variance))
            
            if uid_variance_values:
                lowest_variance_trace = min(uid_variance_values, key=lambda x: x[1])
                traces_to_remove.add(lowest_variance_trace[0])
        
        # 3) Filter out trace with highest number of spikes and falls
        if any('combined_count' in output for output in outputs):
            spike_fall_values = []
            for output in outputs:
                if 'combined_count' in output:
                    spike_fall_values.append((output['index'], output['combined_count']))
            
            if spike_fall_values:
                highest_spike_fall_trace = max(spike_fall_values, key=lambda x: x[1])
                traces_to_remove.add(highest_spike_fall_trace[0])
        
        # 4) Filter out trace with lowest number of spikes and falls
        if any('combined_count' in output for output in outputs):
            spike_fall_values = []
            for output in outputs:
                if 'combined_count' in output:
                    spike_fall_values.append((output['index'], output['combined_count']))
            
            if spike_fall_values:
                lowest_spike_fall_trace = min(spike_fall_values, key=lambda x: x[1])
                traces_to_remove.add(lowest_spike_fall_trace[0])
        
        # Remove the identified traces from the filtered problem
        for trace_index in traces_to_remove:
            # Remove all fields associated with this trace
            keys_to_remove = []
            for key in filtered_problem.keys():
                if key.endswith(f'_{trace_index}') or key == f'Output_{trace_index}' or key == f'Metrics_{trace_index}' or key == f'id_metrics_{trace_index}_metrics' or key == f'id_h_{trace_index}':
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del filtered_problem[key]
        
        # Only keep the problem if it still has at least one trace
        remaining_traces = 0
        i = 0
        while f'Output_{i}' in filtered_problem:
            remaining_traces += 1
            i += 1
        
        if remaining_traces > 0:
            filtered_data.append(filtered_problem)
    
    return filtered_data


def select_traces_by_spikes_falls(data, threshold_pos, threshold_neg):
    """
    Select specific traces for each question based on spike/fall counts:
    1) Highest combined spike+fall count
    2) Lowest combined spike+fall count

    Returns data with only the selected traces for each question, marked with selection reason.
    """
    selected_data = []

    for problem in data:
        problem_id = problem.get('id', 'unknown')

        # Find all outputs for this problem
        outputs = []
        i = 0
        while f'Output_{i}' in problem:
            output_key = f'Output_{i}'
            metrics_key = f'Metrics_{i}'
            uid_metrics_key = f'id_metrics_{i}_metrics'
            id_h_key = f'id_h_{i}'

            output_data = {
                'index': i,
                'output_key': output_key,
                'metrics_key': metrics_key,
                'uid_metrics_key': uid_metrics_key,
                'id_h_key': id_h_key
            }

            # Get UID metrics if available
            if uid_metrics_key in problem:
                output_data['uid_metrics'] = problem[uid_metrics_key]

            # Get id_h data for spike/fall calculation
            if id_h_key in problem:
                output_data['id_h'] = problem[id_h_key]

            outputs.append(output_data)
            i += 1

        if len(outputs) <= 1:
            # If only one or no outputs, keep the problem as is
            selected_data.append(problem)
            continue

        # Create a copy of the problem with only basic fields
        selected_problem = {
            'id': problem.get('id'),
            'year': problem.get('year'),
            'problem_number': problem.get('problem_number'),
            'Question': problem.get('Question'),
            'answer': problem.get('answer'),
            'per_question_mean_accuracy': problem.get('per_question_mean_accuracy'),
            'per_question_mean_validity': problem.get('per_question_mean_validity')
        }

        # Find traces to select based on spike/fall counts
        traces_to_select = {}

        # Collect spike/fall counts for each output
        spike_fall_values = []
        for output in outputs:
            if 'id_h' in output and output['id_h'] and len(output['id_h']) >= 2:
                spike_fall_counts = count_spikes_falls(output['id_h'], threshold_pos, threshold_neg)
                combined_count = spike_fall_counts['spikes'] + spike_fall_counts['falls']
                spike_fall_values.append((output['index'], combined_count, spike_fall_counts))

        if spike_fall_values:
            # Find highest combined spike+fall count
            highest_spike_fall_trace = max(spike_fall_values, key=lambda x: x[1])
            traces_to_select[highest_spike_fall_trace[0]] = {
                'reason': 'highest_spikes_falls',
                'value': highest_spike_fall_trace[1],
                'spike_fall_counts': highest_spike_fall_trace[2]
            }

            # Find lowest combined spike+fall count
            lowest_spike_fall_trace = min(spike_fall_values, key=lambda x: x[1])
            traces_to_select[lowest_spike_fall_trace[0]] = {
                'reason': 'lowest_spikes_falls',
                'value': lowest_spike_fall_trace[1],
                'spike_fall_counts': lowest_spike_fall_trace[2]
            }

        # Add only the selected traces to the selected problem with marking and renamed keys
        for trace_index, selection_info in traces_to_select.items():
            # Determine the new trace index based on selection reason
            if selection_info['reason'] == 'highest_spikes_falls':
                new_trace_index = 'highest'
            else:  # lowest_spikes_falls
                new_trace_index = 'lowest'

            # Add all fields associated with this trace with renamed keys
            for key in problem.keys():
                if key.endswith(f'_{trace_index}') or key == f'Output_{trace_index}' or key == f'Metrics_{trace_index}' or key == f'id_metrics_{trace_index}_metrics' or key == f'id_h_{trace_index}':
                    # Rename the key to use the new trace index
                    if key == f'Output_{trace_index}':
                        new_key = f'Output_{new_trace_index}'
                    elif key == f'Metrics_{trace_index}':
                        new_key = f'Metrics_{new_trace_index}'
                    elif key == f'id_metrics_{trace_index}_metrics':
                        new_key = f'id_metrics_{new_trace_index}_metrics'
                    elif key == f'id_h_{trace_index}':
                        new_key = f'id_h_{new_trace_index}'
                    else:
                        # For other keys ending with trace_index, replace with new_trace_index
                        new_key = key.replace(f'_{trace_index}', f'_{new_trace_index}')

                    selected_problem[new_key] = problem[key]

        # Add selection metadata
        selected_problem['selection_method'] = 'spikes_falls'
        selected_problem['selection_threshold_pos'] = threshold_pos
        selected_problem['selection_threshold_neg'] = threshold_neg

        # Add selection details for each trace
        for trace_index, selection_info in traces_to_select.items():
            if selection_info['reason'] == 'highest_spikes_falls':
                new_trace_index = 'highest'
            else:
                new_trace_index = 'lowest'

            selected_problem[f'selection_reason_{new_trace_index}'] = selection_info['reason']
            selected_problem[f'selection_value_{new_trace_index}'] = selection_info['value']
            selected_problem[f'spikes_count_{new_trace_index}'] = selection_info['spike_fall_counts']['spikes']
            selected_problem[f'falls_count_{new_trace_index}'] = selection_info['spike_fall_counts']['falls']

        selected_data.append(selected_problem)

    return selected_data


def select_traces_by_criteria(data, avg_positive=None, avg_negative=None):
    """
    Select specific traces for each question based on uid_variance_entropy:
    1) Highest uid_variance_entropy
    2) Lowest uid_variance_entropy
    
    Returns data with only the selected traces for each question, marked with selection reason.
    """
    selected_data = []
    
    for problem in data:
        problem_id = problem.get('id', 'unknown')
        
        # Find all outputs for this problem
        outputs = []
        i = 0
        while f'Output_{i}' in problem:
            output_key = f'Output_{i}'
            metrics_key = f'Metrics_{i}'
            uid_metrics_key = f'id_metrics_{i}_metrics'
            
            output_data = {
                'index': i,
                'output_key': output_key,
                'metrics_key': metrics_key,
                'uid_metrics_key': uid_metrics_key
            }
            
            # Get UID metrics if available
            if uid_metrics_key in problem:
                output_data['uid_metrics'] = problem[uid_metrics_key]
            
            outputs.append(output_data)
            i += 1
        
        if len(outputs) <= 1:
            # If only one or no outputs, keep the problem as is
            selected_data.append(problem)
            continue
        
        # Create a copy of the problem with only basic fields
        selected_problem = {
            'id': problem.get('id'),
            'year': problem.get('year'),
            'problem_number': problem.get('problem_number'),
            'Question': problem.get('Question'),
            'answer': problem.get('answer'),
            'per_question_mean_accuracy': problem.get('per_question_mean_accuracy'),
            'per_question_mean_validity': problem.get('per_question_mean_validity')
        }
        
        # Find traces to select based on uid_variance_entropy
        traces_to_select = {}
        
        # Collect uid_variance_entropy values
        if any('uid_metrics' in output for output in outputs):
            uid_variance_entropy_values = []
            nan_count = 0
            for output in outputs:
                if 'uid_metrics' in output:
                    uid_variance_entropy = output['uid_metrics'].get('uid_variance_entropy', 0)
                    # Skip NaN values
                    if isinstance(uid_variance_entropy, float) and np.isnan(uid_variance_entropy):
                        nan_count += 1
                    else:
                        uid_variance_entropy_values.append((output['index'], uid_variance_entropy))
            
            if nan_count > 0:
                print(f"Warning: Found {nan_count} NaN uid_variance_entropy values for problem {problem_id}")
            
            if uid_variance_entropy_values:
                # Find highest uid_variance_entropy
                highest_variance_entropy_trace = max(uid_variance_entropy_values, key=lambda x: x[1])
                traces_to_select[highest_variance_entropy_trace[0]] = {
                    'reason': 'highest_uid_variance_entropy',
                    'value': highest_variance_entropy_trace[1]
                }
                
                # Find lowest uid_variance_entropy
                lowest_variance_entropy_trace = min(uid_variance_entropy_values, key=lambda x: x[1])
                traces_to_select[lowest_variance_entropy_trace[0]] = {
                    'reason': 'lowest_uid_variance_entropy',
                    'value': lowest_variance_entropy_trace[1]
                }
            else:
                print(f"Warning: No valid uid_variance_entropy values found for problem {problem_id}")
        
        # Add only the selected traces to the selected problem with marking and renamed keys
        for trace_index, selection_info in traces_to_select.items():
            # Determine the new trace index based on selection reason
            if selection_info['reason'] == 'highest_uid_variance_entropy':
                new_trace_index = 'highest'
            else:  # lowest_uid_variance_entropy
                new_trace_index = 'lowest'
            
            # Add all fields associated with this trace with renamed keys
            for key in problem.keys():
                if key.endswith(f'_{trace_index}') or key == f'Output_{trace_index}' or key == f'Metrics_{trace_index}' or key == f'id_metrics_{trace_index}_metrics' or key == f'id_h_{trace_index}':
                    # Rename the key to use the new trace index
                    if key == f'Output_{trace_index}':
                        new_key = f'Output_{new_trace_index}'
                    elif key == f'Metrics_{trace_index}':
                        new_key = f'Metrics_{new_trace_index}'
                    elif key == f'id_metrics_{trace_index}_metrics':
                        new_key = f'id_metrics_{new_trace_index}_metrics'
                    elif key == f'id_h_{trace_index}':
                        new_key = f'id_h_{new_trace_index}'
                    else:
                        # For other keys ending with trace_index, replace with new_trace_index
                        new_key = key.replace(f'_{trace_index}', f'_{new_trace_index}')
                    
                    selected_problem[new_key] = problem[key]
            
            # Add selection metadata with new trace index
            selected_problem[f'selection_reason_{new_trace_index}'] = selection_info['reason']
            selected_problem[f'uid_variance_entropy_value_{new_trace_index}'] = selection_info['value']
        
        # Add summary of selections for this problem
        selected_problem['trace_selections'] = {
            'highest' if info['reason'] == 'highest_uid_variance_entropy' else 'lowest': {
                'reason': info['reason'],
                'uid_variance_entropy': info['value'],
                'original_trace_index': trace_idx
            }
            for trace_idx, info in traces_to_select.items()
        }
        
        # Only keep the problem if it has at least one trace
        remaining_traces = 0
        # Check for both 'highest' and 'lowest' traces
        if 'Output_highest' in selected_problem:
            remaining_traces += 1
        if 'Output_lowest' in selected_problem:
            remaining_traces += 1
        
        if remaining_traces > 0:
            selected_data.append(selected_problem)
    
    return selected_data


def analyze_spikes_falls_accuracy_by_level(data, avg_positive, avg_negative):
    """
    Group problems by 'level', then for each level compute spikes/falls analysis
    using the same logic as analyze_spikes_falls_accuracy.
    """
    # Group problems by level
    problems_by_level = defaultdict(list)
    for problem in data:
        level = problem.get('level', 'unknown')
        problems_by_level[level].append(problem)

    level_results = {}
    for level, level_problems in problems_by_level.items():
        level_results[level] = analyze_spikes_falls_accuracy(level_problems, avg_positive, avg_negative)
    return level_results


def create_spikes_falls_level_boxplots(level_spikes_falls_summary, outdir, filename_prefix, input_filename=None, overall_metrics=None):
    """
    For each level, plot mean accuracy for Highest Spikes+Falls vs Lowest Spikes+Falls.
    Also produce a combined multi-level figure.
    """
    plt.style.use('default')
    sns.set_palette("husl")

    levels = sorted(level_spikes_falls_summary.keys(), key=lambda x: str(x))
    fig, axes = plt.subplots(len(levels), 1, figsize=(16, 6 * max(len(levels), 1)))
    if len(levels) == 1:
        axes = [axes]

    if input_filename:
        dataset_name = input_filename.split("/")[-2] if "/" in input_filename else input_filename.replace(".json", "")
        suptitle = 'Spike/Fall Analysis by Level: Mean Accuracy \n ' + dataset_name
    else:
        suptitle = 'Spike/Fall Analysis by Level: Mean Accuracy \n ' + filename_prefix
    fig.suptitle(suptitle, fontsize=16, fontweight='bold')

    combined_plot_files = []
    for i, level in enumerate(levels):
        ax = axes[i]
        summary = level_spikes_falls_summary[level]

        # Pull one metric representative (we only have 'combined')
        def get_mean_acc(analysis_type):
            if analysis_type in summary:
                for metric in summary[analysis_type]:
                    return summary[analysis_type][metric]['mean_accuracy']
            return 0.0

        labels = ['Highest Spikes+Falls', 'Lowest Spikes+Falls']
        values = [get_mean_acc('highest_spikes_falls'), get_mean_acc('lowest_spikes_falls')]

        x = np.arange(len(labels))
        width = 0.6
        bars = ax.bar(x, values, width, color=['lightblue', 'lightcoral'], alpha=0.8)

        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        if overall_metrics and isinstance(overall_metrics, dict):
            overall_accuracy = overall_metrics.get('overall_mean_accuracy', None)
            if overall_accuracy is not None:
                ax.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Acc ({overall_accuracy:.3f})', alpha=0.8)

        ax.set_title(f'Level {level} - Mean Accuracy (Spikes/Falls)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Accuracy')
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

        # Individual per-level plot
        plt.figure(figsize=(12, 8))
        ind_ax = plt.gca()
        ind_bars = ind_ax.bar(x, values, width, color=['lightblue', 'lightcoral'], alpha=0.8)
        for bar, value in zip(ind_bars, values):
            ind_ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, f'{value:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=9)

        if overall_metrics and isinstance(overall_metrics, dict):
            overall_accuracy = overall_metrics.get('overall_mean_accuracy', None)
            if overall_accuracy is not None:
                ind_ax.axhline(y=overall_accuracy, color='red', linestyle='--', linewidth=2, label=f'Overall Acc ({overall_accuracy:.3f})', alpha=0.8)

        ind_ax.set_title(f'Level {level} - Mean Accuracy (Spikes/Falls)')
        ind_ax.set_ylabel('Mean Accuracy')
        ind_ax.set_xticks(x)
        ind_ax.set_xticklabels(labels, rotation=45, ha='right')
        ind_ax.legend()
        ind_ax.grid(True, alpha=0.3)
        ind_ax.set_ylim(0, 1.1)

        per_level_plot = os.path.join(outdir, f"{filename_prefix}_spikes_falls_level_{level}.png")
        plt.savefig(per_level_plot, dpi=300, bbox_inches='tight')
        plt.close()
        combined_plot_files.append(per_level_plot)

    plt.tight_layout()
    combined_plot = os.path.join(outdir, f"{filename_prefix}_spikes_falls_analysis_by_level.png")
    plt.savefig(combined_plot, dpi=300, bbox_inches='tight')
    plt.close()

    return combined_plot, combined_plot_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--input2", type=str, required=False, help="Path to metrics JSON file containing overall_mean_accuracy and overall_mean_self_certainty_accuracy")
    parser.add_argument("--outdir", default="analysis_out")
    parser.add_argument("--analysis_by_level", action='store_true', help="Enable analysis by difficulty level")
    parser.add_argument("--analysis_by_domain", action='store_true', help="Enable analysis by domain")
    parser.add_argument("--select_traces", action='store_true', help="Select traces with highest/lowest uid_variance_entropy for each question")
    parser.add_argument("--select_traces_by_spikes_falls", action='store_true', help="Select traces with highest/lowest spikes and falls for each question")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.input, "r") as f:
        data = json.load(f)
    
    # Load overall metrics if provided
    overall_metrics = None
    level_specific_metrics = None
    if args.input2:
        with open(args.input2, "r") as f:
            metrics_data = json.load(f)
            # Extract metrics from the "overall" key if it exists
            if "overall" in metrics_data:
                overall_metrics = metrics_data["overall"]
            else:
                overall_metrics = metrics_data
            
            # Extract level-specific metrics from "per_domain" key
            if "per_domain" in metrics_data:
                level_specific_metrics = metrics_data["per_domain"]
    
    print(f"Analyzing {len(data)} problems...")
    
    # Store original data for spike/fall analysis if both selections are requested
    original_data = data.copy() if args.select_traces and args.select_traces_by_spikes_falls else None
    
    # Apply trace selection if requested
    if args.select_traces:
        print("\n" + "="*80)
        print("APPLYING UID VARIANCE ENTROPY TRACE SELECTION")
        print("="*80)
        print("Selection criteria: Highest and lowest uid_variance_entropy for each question")
        
        # Apply selection (no need for spike/fall thresholds since we're only using UID metrics)
        original_count = len(data)
        data = select_traces_by_criteria(data)
        selected_count = len(data)
        
        print(f"Original problems: {original_count}")
        print(f"Problems after selection: {selected_count}")
        
        # Count traces after selection (now using 'highest' and 'lowest' naming)
        selected_traces = 0
        for problem in data:
            if 'Output_highest' in problem:
                selected_traces += 1
            if 'Output_lowest' in problem:
                selected_traces += 1
        print(f"Total traces after selection: {selected_traces}")
        
        # Print selection summary
        total_highest = 0
        total_lowest = 0
        for problem in data:
            if 'trace_selections' in problem:
                if 'highest' in problem['trace_selections']:
                    total_highest += 1
                if 'lowest' in problem['trace_selections']:
                    total_lowest += 1
        
        print(f"Traces selected for highest uid_variance_entropy: {total_highest}")
        print(f"Traces selected for lowest uid_variance_entropy: {total_lowest}")
        
        # Save selected data
        selected_data_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_selected_data_uid.json")
        with open(selected_data_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"UID-based selected data saved to: {selected_data_file}")
    
    if args.select_traces_by_spikes_falls:
        print("\n" + "="*80)
        print("APPLYING SPIKE/FALL TRACE SELECTION")
        print("="*80)
        print("Selection criteria: Highest and lowest combined spike+fall counts for each question")
        print("Using 3-sigma thresholds for spike/fall detection")
        
        # Use original data if both selections are requested (to avoid field naming conflicts)
        spike_fall_data = original_data if original_data is not None else data
        
        # Calculate thresholds first
        avg_positive, avg_negative, threshold_2sigma_pos, threshold_2sigma_neg, threshold_3sigma_pos, threshold_3sigma_neg = calculate_sigma_thresholds(spike_fall_data)
        
        # Apply selection using 3-sigma thresholds
        original_count = len(spike_fall_data)
        spike_fall_selected_data = select_traces_by_spikes_falls(spike_fall_data, threshold_3sigma_pos, threshold_3sigma_neg)
        selected_count = len(spike_fall_selected_data)
        
        print(f"Original problems: {original_count}")
        print(f"Problems after selection: {selected_count}")
        print(f"Using 3-sigma thresholds: pos={threshold_3sigma_pos:.4f}, neg={threshold_3sigma_neg:.4f}")
        
        # Count traces after selection (now using 'highest' and 'lowest' naming)
        selected_traces = 0
        for problem in spike_fall_selected_data:
            if 'Output_highest' in problem:
                selected_traces += 1
            if 'Output_lowest' in problem:
                selected_traces += 1
        
        print(f"Total selected traces: {selected_traces}")
        
        # Print selection summary
        total_highest = 0
        total_lowest = 0
        for problem in spike_fall_selected_data:
            if 'selection_reason_highest' in problem:
                total_highest += 1
            if 'selection_reason_lowest' in problem:
                total_lowest += 1
        
        print(f"Traces selected for highest spikes+falls: {total_highest}")
        print(f"Traces selected for lowest spikes+falls: {total_lowest}")
        
        # Save selected data
        selected_data_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_selected_data_spikes_falls.json")
        with open(selected_data_file, "w") as f:
            json.dump(spike_fall_selected_data, f, indent=2)
        print(f"Spike/fall-based selected data saved to: {selected_data_file}")
        print("="*80)
    
    # Print overall metrics for debugging
    if overall_metrics:
        print(f"Overall metrics loaded: {overall_metrics}")
        print(f"Overall accuracy: {overall_metrics.get('overall_mean_accuracy', 'Not found')}")
        print(f"Overall self-certainty: {overall_metrics.get('overall_mean_self_certainty_accuracy', 'Not found')}")
        print(f"Overall cot-decoding: {overall_metrics.get('overall_mean_cot_decoding_accuracy', 'Not found')}")
        print(f"Overall confidence: {overall_metrics.get('overall_mean_confidence_accuracy', 'Not found')}")
        print(f"Overall entropy: {overall_metrics.get('overall_mean_entropy_accuracy', 'Not found')}")
    # Print level-specific metrics for debugging
    if level_specific_metrics:
        print(f"Level-specific metrics loaded: {level_specific_metrics}")
    
    # Analyze UID-based accuracy
    if args.select_traces:
        # Use the new function for selected data
        results = analyze_selected_uid_accuracy(data)
    else:
        # Use the original function for unselected data
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
                  f"F1: {stats['mean_f1']:.4f} ± {stats['mean_f1']:.4f} | "
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
        print(f"Mean Exact Match: {stats['mean_exact_match']:.4f} ± {stats['mean_exact_match_std']:.4f}")
        print(f"Mean Exact Match Std: {stats['mean_exact_match_std']:.4f} ± {stats['std_exact_match_std']:.4f}")
        print(f"Mean F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f}")
        print(f"Mean F1 Std: {stats['mean_f1_std']:.4f} ± {stats['std_f1_std']:.4f}")
        print(f"Mean UID Score: {stats['mean_uid_score']:.4f} ± {stats['std_uid_score']:.4f}")
        print(f"Mean UID Score Std: {stats['mean_uid_score_std']:.4f} ± {stats['std_uid_score_std']:.4f}")
    
    # Create boxplots
    filename_prefix = args.input.split("/")[-2].replace("outputs_old", "outputs") if "/" in args.input else args.input.replace(".json", "").replace("outputs_old", "outputs")
    plot_file = create_boxplots(summary, results, args.outdir, filename_prefix, args.input, overall_metrics)
    
    # Create correlation plots
    correlation_plot_file = create_correlation_plot(data, args.outdir, filename_prefix, args.input)
    
    # Extract Shannon Equal outputs
    shannon_file = extract_shannon_equal_outputs(data, args.outdir, filename_prefix)
    
    # Analyze spike/fall-based accuracy
    print("\n" + "="*80)
    print("SPIKE/FALL ANALYSIS")
    print("="*80)
    
    # Use original data for spike/fall analysis if both selections were requested
    spike_fall_analysis_data = original_data if original_data is not None else data
    
    # Calculate global thresholds
    avg_positive, avg_negative, threshold_2sigma_pos, threshold_2sigma_neg, threshold_3sigma_pos, threshold_3sigma_neg = calculate_sigma_thresholds(spike_fall_analysis_data)
    print(f"Average positive difference: {avg_positive:.4f}")
    print(f"Average negative difference: {avg_negative:.4f}")
    print(f"2-sigma thresholds: pos={threshold_2sigma_pos:.4f}, neg={threshold_2sigma_neg:.4f}")
    print(f"3-sigma thresholds: pos={threshold_3sigma_pos:.4f}, neg={threshold_3sigma_neg:.4f}")
    
    # Analyze spikes and falls
    spikes_falls_results = analyze_spikes_falls_accuracy(spike_fall_analysis_data, avg_positive, avg_negative)
    spikes_falls_results_2sigma = analyze_spikes_falls_accuracy(spike_fall_analysis_data, threshold_2sigma_pos, threshold_2sigma_neg)
    spikes_falls_results_3sigma = analyze_spikes_falls_accuracy(spike_fall_analysis_data, threshold_3sigma_pos, threshold_3sigma_neg)
    
    # Calculate summary statistics for spikes/falls
    spikes_falls_summary = calculate_spikes_falls_summary_stats(spikes_falls_results)
    spikes_falls_summary_2sigma = calculate_spikes_falls_summary_stats(spikes_falls_results_2sigma)
    spikes_falls_summary_3sigma = calculate_spikes_falls_summary_stats(spikes_falls_results_3sigma)
    
    # Print spike/fall results
    print("\n" + "="*80)
    print("ACCURACY ANALYSIS BASED ON COMBINED SPIKE/FALL COUNTS")
    print("="*80)
    
    for analysis_type in ['highest_spikes_falls', 'lowest_spikes_falls']:
        print(f"\n{analysis_type.upper().replace('_', ' ')}:")
        print("-" * 60)
        
        for metric in spikes_falls_summary[analysis_type]:
            stats = spikes_falls_summary[analysis_type][metric]
            print(f"{metric:35} | Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} | "
                  f"Exact Match: {stats['mean_exact_match']:.4f} ± {stats['std_exact_match']:.4f} | "
                  f"F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f} | "
                  f"Combined Count: {stats['mean_combined_count']:.2f} ± {stats['std_combined_count']:.2f} | "
                  f"Spikes: {stats['mean_spikes_count']:.2f} ± {stats['std_spikes_count']:.2f} | "
                  f"Falls: {stats['mean_falls_count']:.2f} ± {stats['std_falls_count']:.2f}")
    
    # Create spike/fall boxplots
    spikes_falls_plot_file = create_spikes_falls_boxplots(spikes_falls_summary, args.outdir, filename_prefix, args.input, overall_metrics)
    
    # Save spike/fall results
    spikes_falls_results_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_spikes_falls_analysis.json")
    with open(spikes_falls_results_file, "w") as f:
        json.dump(spikes_falls_results, f, indent=2)
    spikes_falls_results_file_2sigma = os.path.join(args.outdir, args.input.split("/")[-2] + "_spikes_falls_analysis_2sigma.json")
    with open(spikes_falls_results_file_2sigma, "w") as f:
        json.dump(spikes_falls_results_2sigma, f, indent=2)
    spikes_falls_results_file_3sigma = os.path.join(args.outdir, args.input.split("/")[-2] + "_spikes_falls_analysis_3sigma.json")
    with open(spikes_falls_results_file_3sigma, "w") as f:
        json.dump(spikes_falls_results_3sigma, f, indent=2)
    
    # Save spike/fall summary
    spikes_falls_summary_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_spikes_falls_summary.json")
    with open(spikes_falls_summary_file, "w") as f:
        json.dump(spikes_falls_summary, f, indent=2)
    spikes_falls_summary_file_2sigma = os.path.join(args.outdir, args.input.split("/")[-2] + "_spikes_falls_summary_2sigma.json")
    with open(spikes_falls_summary_file_2sigma, "w") as f:
        json.dump(spikes_falls_summary_2sigma, f, indent=2)
    spikes_falls_summary_file_3sigma = os.path.join(args.outdir, args.input.split("/")[-2] + "_spikes_falls_summary_3sigma.json")
    with open(spikes_falls_summary_file_3sigma, "w") as f:
        json.dump(spikes_falls_summary_3sigma, f, indent=2)
    
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
    
    # Analysis by level if requested
    if args.analysis_by_level:
        print("\n" + "="*80)
        print("ANALYSIS BY DIFFICULTY LEVEL")
        print("="*80)
        
        # Analyze UID-based accuracy by level
        level_results = analyze_uid_accuracy_by_level(data)
        
        # Calculate summary statistics for each level
        level_summary = {}
        for level, level_data in level_results.items():
            level_summary[level] = calculate_summary_stats(level_data)
        
        # Calculate aggregated statistics for each level
        level_aggregated = {}
        for level, level_data in level_results.items():
            # We need to pass the summary, not the raw results
            level_aggregated[level] = calculate_aggregated_stats(level_summary[level])
        
        # Print results by level
        for level in sorted(level_summary.keys()):
            print(f"\nLEVEL {level} - ACCURACY ANALYSIS BASED ON UID SCORES:")
            print("-" * 60)
            
            for uid_type in ['highest_uid', 'lowest_uid']:
                print(f"\n{uid_type.upper().replace('_', ' ')} UID SCORES:")
                print("-" * 40)
                
                for metric in sorted(level_summary[level][uid_type].keys()):
                    stats = level_summary[level][uid_type][metric]
                    print(f"{metric:35} | Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} | "
                          f"Exact Match: {stats['mean_exact_match']:.4f} ± {stats['std_exact_match']:.4f} | "
                          f"F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f} | "
                          f"UID Score: {stats['mean_uid_score']:.4f} ± {stats['std_uid_score']:.4f}")
            
            # Print aggregated results for this level
            print(f"\nLEVEL {level} - AGGREGATED STATISTICS:")
            print("-" * 40)
            stats = level_aggregated[level]
            for uid_type in ['highest_uid', 'lowest_uid']:
                print(f"\n{uid_type.upper().replace('_', ' ')} UID SCORES (AGGREGATED):")
                print("-" * 30)
                level_stats = stats[uid_type]
                print(f"Mean Accuracy: {level_stats['mean_accuracy']:.4f} ± {level_stats['std_accuracy']:.4f}")
                print(f"Mean Accuracy Std: {level_stats['mean_accuracy_std']:.4f} ± {level_stats['std_accuracy_std']:.4f}")
                print(f"Mean Exact Match: {level_stats['mean_exact_match']:.4f} ± {level_stats['mean_exact_match_std']:.4f}")
                print(f"Mean Exact Match Std: {level_stats['mean_exact_match_std']:.4f} ± {level_stats['std_exact_match_std']:.4f}")
                print(f"Mean F1: {level_stats['mean_f1']:.4f} ± {level_stats['std_f1']:.4f}")
                print(f"Mean F1 Std: {level_stats['mean_f1_std']:.4f} ± {level_stats['std_f1_std']:.4f}")
                print(f"Mean UID Score: {level_stats['mean_uid_score']:.4f} ± {level_stats['std_uid_score']:.4f}")
                print(f"Mean UID Score Std: {level_stats['mean_uid_score_std']:.4f} ± {level_stats['std_uid_score_std']:.4f}")
        
        # Create level-specific boxplots with level-specific metrics
        level_plot_file, individual_level_plots = create_level_boxplots(level_summary, level_results, args.outdir, filename_prefix, args.input, overall_metrics, level_specific_metrics)
        
        # Save level-specific results
        level_results_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_analysis_by_level.json")
        with open(level_results_file, "w") as f:
            json.dump(level_results, f, indent=2)
        
        # Save level-specific summary
        level_summary_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_summary_by_level.json")
        with open(level_summary_file, "w") as f:
            json.dump(level_summary, f, indent=2)
        
        # Save level-specific aggregated results
        level_aggregated_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_aggregated_by_level.json")
        with open(level_aggregated_file, "w") as f:
            json.dump(level_aggregated, f, indent=2)
        
        print(f"\nLevel-specific detailed results saved to: {level_results_file}")
        print(f"Level-specific summary saved to: {level_summary_file}")
        print(f"Level-specific aggregated results saved to: {level_aggregated_file}")
        print(f"Level-specific combined boxplots saved to: {level_plot_file}")
        print(f"Individual level plots saved to:")
        for plot_file in individual_level_plots:
            print(f"  - {plot_file}")
    
    # Analysis by domain if requested
    if args.analysis_by_domain:
        print("\n" + "="*80)
        print("ANALYSIS BY HIGH-LEVEL DOMAIN")
        print("="*80)
        
        # Analyze UID-based accuracy by domain
        domain_results = analyze_uid_accuracy_by_domain(data)
        
        # Calculate summary statistics for each domain
        domain_summary = {}
        for domain, domain_data in domain_results.items():
            domain_summary[domain] = calculate_summary_stats(domain_data)
        
        # Calculate aggregated statistics for each domain
        domain_aggregated = {}
        for domain, domain_data in domain_results.items():
            # We need to pass the summary, not the raw results
            domain_aggregated[domain] = calculate_aggregated_stats(domain_summary[domain])
        
        # Print results by domain
        for domain in sorted(domain_summary.keys()):
            print(f"\nDOMAIN {domain} - ACCURACY ANALYSIS BASED ON UID SCORES:")
            print("-" * 60)
            
            for uid_type in ['highest_uid', 'lowest_uid']:
                print(f"\n{uid_type.upper().replace('_', ' ')} UID SCORES:")
                print("-" * 40)
                
                for metric in sorted(domain_summary[domain][uid_type].keys()):
                    stats = domain_summary[domain][uid_type][metric]
                    print(f"{metric:35} | Accuracy: {stats['mean_accuracy']:.4f} ± {stats['std_accuracy']:.4f} | "
                          f"Exact Match: {stats['mean_exact_match']:.4f} ± {stats['std_exact_match']:.4f} | "
                          f"F1: {stats['mean_f1']:.4f} ± {stats['std_f1']:.4f} | "
                          f"UID Score: {stats['mean_uid_score']:.4f} ± {stats['std_uid_score']:.4f}")
            
            # Print aggregated results for this domain
            print(f"\nDOMAIN {domain} - AGGREGATED STATISTICS:")
            print("-" * 40)
            stats = domain_aggregated[domain]
            for uid_type in ['highest_uid', 'lowest_uid']:
                print(f"\n{uid_type.upper().replace('_', ' ')} UID SCORES (AGGREGATED):")
                print("-" * 30)
                domain_stats = stats[uid_type]
                print(f"Mean Accuracy: {domain_stats['mean_accuracy']:.4f} ± {domain_stats['std_accuracy']:.4f}")
                print(f"Mean Accuracy Std: {domain_stats['mean_accuracy_std']:.4f} ± {domain_stats['std_accuracy_std']:.4f}")
                print(f"Mean Exact Match: {domain_stats['mean_exact_match']:.4f} ± {domain_stats['mean_exact_match_std']:.4f}")
                print(f"Mean Exact Match Std: {domain_stats['mean_exact_match_std']:.4f} ± {domain_stats['std_exact_match_std']:.4f}")
                print(f"Mean F1: {domain_stats['mean_f1']:.4f} ± {domain_stats['std_f1']:.4f}")
                print(f"Mean F1 Std: {domain_stats['mean_f1_std']:.4f} ± {domain_stats['std_f1_std']:.4f}")
                print(f"Mean UID Score: {domain_stats['mean_uid_score']:.4f} ± {domain_stats['std_uid_score']:.4f}")
                print(f"Mean UID Score Std: {domain_stats['mean_uid_score_std']:.4f} ± {domain_stats['std_uid_score_std']:.4f}")
        
        # Create domain-specific boxplots
        domain_plot_file, individual_domain_plots = create_domain_boxplots(domain_summary, domain_results, args.outdir, filename_prefix, args.input, overall_metrics)
        
        # Save domain-specific results
        domain_results_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_analysis_by_domain.json")
        with open(domain_results_file, "w") as f:
            json.dump(domain_results, f, indent=2)
        
        # Save domain-specific summary
        domain_summary_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_summary_by_domain.json")
        with open(domain_summary_file, "w") as f:
            json.dump(domain_summary, f, indent=2)
        
        # Save domain-specific aggregated results
        domain_aggregated_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_uid_aggregated_by_domain.json")
        with open(domain_aggregated_file, "w") as f:
            json.dump(domain_aggregated, f, indent=2)
        
        print(f"\nDomain-specific detailed results saved to: {domain_results_file}")
        print(f"Domain-specific summary saved to: {domain_summary_file}")
        print(f"Domain-specific aggregated results saved to: {domain_aggregated_file}")
        print(f"Domain-specific combined boxplots saved to: {domain_plot_file}")
        print(f"Individual domain plots saved to:")
        for plot_file in individual_domain_plots:
            print(f"  - {plot_file}")
    
    print(f"\nDetailed results saved to: {results_file}")
    print(f"Summary saved to: {summary_file}")
    print(f"Aggregated results saved to: {aggregated_file}")
    print(f"Combined analysis plot saved to: {plot_file}")
    print(f"Spike/Fall analysis (average thresholds) saved to: {spikes_falls_results_file}")
    print(f"Spike/Fall summary (average thresholds) saved to: {spikes_falls_summary_file}")
    print(f"Spike/Fall analysis (2-sigma thresholds) saved to: {spikes_falls_results_file_2sigma}")
    print(f"Spike/Fall summary (2-sigma thresholds) saved to: {spikes_falls_summary_file_2sigma}")
    print(f"Spike/Fall analysis (3-sigma thresholds) saved to: {spikes_falls_results_file_3sigma}")
    print(f"Spike/Fall summary (3-sigma thresholds) saved to: {spikes_falls_summary_file_3sigma}")
    if correlation_plot_file:
        print(f"Correlation plots saved to: {correlation_plot_file}")
    if shannon_file:
        print(f"Shannon Equal outputs saved to: {shannon_file}")
    
    # Create combined analysis plot
    combined_plot_file = create_combined_analysis_plot(summary, spikes_falls_summary, spikes_falls_summary_2sigma, spikes_falls_summary_3sigma, args.outdir, filename_prefix, args.input, overall_metrics)
    print(f"Combined analysis plot saved to: {combined_plot_file}")

    # Analysis by level if requested
    if args.analysis_by_level:
        # Per-level spikes/falls analysis
        level_spikes_falls_results = analyze_spikes_falls_accuracy_by_level(data, avg_positive, avg_negative)

        # Summaries per level
        level_spikes_falls_summary = {}
        for level, res in level_spikes_falls_results.items():
            level_spikes_falls_summary[level] = calculate_spikes_falls_summary_stats(res)

        # Plots per level
        level_spikes_falls_plot, per_level_spikes_falls_plots = create_spikes_falls_level_boxplots(
            level_spikes_falls_summary, args.outdir, filename_prefix, args.input, overall_metrics
        )

        # Save per-level spikes/falls results and summaries
        spikes_falls_by_level_results_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_spikes_falls_analysis_by_level.json")
        with open(spikes_falls_by_level_results_file, "w") as f:
            json.dump(level_spikes_falls_results, f, indent=2)

        spikes_falls_by_level_summary_file = os.path.join(args.outdir, args.input.split("/")[-2] + "_spikes_falls_summary_by_level.json")
        with open(spikes_falls_by_level_summary_file, "w") as f:
            json.dump(level_spikes_falls_summary, f, indent=2)

        print(f"Per-level spikes/falls analysis saved to: {spikes_falls_by_level_results_file}")
        print(f"Per-level spikes/falls summary saved to: {spikes_falls_by_level_summary_file}")
        print(f"Per-level spikes/falls combined plot saved to: {level_spikes_falls_plot}")
        print("Individual per-level spikes/falls plots:")
        for p in per_level_spikes_falls_plots:
            print(f"  - {p}")

        # Per-level spikes/falls analysis - 2σ thresholds
        level_spikes_falls_results_2sigma = analyze_spikes_falls_accuracy_by_level(
            data, avg_positive + 2 * avg_negative, avg_negative - 2 * avg_positive
        )

        level_spikes_falls_summary_2sigma = {}
        for level, res in level_spikes_falls_results_2sigma.items():
            level_spikes_falls_summary_2sigma[level] = calculate_spikes_falls_summary_stats(res)

        level_spikes_falls_plot_2sigma, per_level_spikes_falls_plots_2sigma = create_spikes_falls_level_boxplots(
            level_spikes_falls_summary_2sigma, args.outdir, filename_prefix + "_2sigma", args.input, overall_metrics
        )

        spikes_falls_by_level_results_file_2sigma = os.path.join(
            args.outdir, args.input.split("/")[-2] + "_spikes_falls_analysis_by_level_2sigma.json"
        )
        with open(spikes_falls_by_level_results_file_2sigma, "w") as f:
            json.dump(level_spikes_falls_results_2sigma, f, indent=2)

        spikes_falls_by_level_summary_file_2sigma = os.path.join(
            args.outdir, args.input.split("/")[-2] + "_spikes_falls_summary_by_level_2sigma.json"
        )
        with open(spikes_falls_by_level_summary_file_2sigma, "w") as f:
            json.dump(level_spikes_falls_summary_2sigma, f, indent=2)

        print(f"Per-level spikes/falls (2σ) analysis saved to: {spikes_falls_by_level_results_file_2sigma}")
        print(f"Per-level spikes/falls (2σ) summary saved to: {spikes_falls_by_level_summary_file_2sigma}")
        print(f"Per-level spikes/falls (2σ) combined plot saved to: {level_spikes_falls_plot_2sigma}")
        for p in per_level_spikes_falls_plots_2sigma:
            print(f"  - {p}")

        # Per-level spikes/falls analysis - 3σ thresholds
        level_spikes_falls_results_3sigma = analyze_spikes_falls_accuracy_by_level(
            data, avg_positive + 3 * avg_negative, avg_negative - 3 * avg_positive
        )

        level_spikes_falls_summary_3sigma = {}
        for level, res in level_spikes_falls_results_3sigma.items():
            level_spikes_falls_summary_3sigma[level] = calculate_spikes_falls_summary_stats(res)

        level_spikes_falls_plot_3sigma, per_level_spikes_falls_plots_3sigma = create_spikes_falls_level_boxplots(
            level_spikes_falls_summary_3sigma, args.outdir, filename_prefix + "_3sigma", args.input, overall_metrics
        )

        spikes_falls_by_level_results_file_3sigma = os.path.join(
            args.outdir, args.input.split("/")[-2] + "_spikes_falls_analysis_by_level_3sigma.json"
        )
        with open(spikes_falls_by_level_results_file_3sigma, "w") as f:
            json.dump(level_spikes_falls_results_3sigma, f, indent=2)

        spikes_falls_by_level_summary_file_3sigma = os.path.join(
            args.outdir, args.input.split("/")[-2] + "_spikes_falls_summary_by_level_3sigma.json"
        )
        with open(spikes_falls_by_level_summary_file_3sigma, "w") as f:
            json.dump(level_spikes_falls_summary_3sigma, f, indent=2)

        print(f"Per-level spikes/falls (3σ) analysis saved to: {spikes_falls_by_level_results_file_3sigma}")
        print(f"Per-level spikes/falls (3σ) summary saved to: {spikes_falls_by_level_summary_file_3sigma}")
        print(f"Per-level spikes/falls (3σ) combined plot saved to: {level_spikes_falls_plot_3sigma}")
        for p in per_level_spikes_falls_plots_3sigma:
            print(f"  - {p}")


if __name__ == "__main__":
    main()