import json
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from scipy.stats import pearsonr


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
            uid_metrics_key = f'uid_metrics_{i}'
            
            if uid_metrics_key in problem and metrics_key in problem:
                uid_data = problem[uid_metrics_key]
                metrics_data = problem[metrics_key]
                
                data_point = {
                    'problem_id': problem_id,
                    'output_index': i,
                    'accuracy': metrics_data.get('acc', 0),
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
        acc_corr, _ = pearsonr(df[metric], df['accuracy'])
        em_corr, _ = pearsonr(df[metric], df['exact_match'])
        f1_corr, _ = pearsonr(df[metric], df['f1'])
        
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
        acc_corr, acc_p = pearsonr(df[metric], df['accuracy'])
        em_corr, em_p = pearsonr(df[metric], df['exact_match'])
        f1_corr, f1_p = pearsonr(df[metric], df['f1'])
        
        print(f"\n{metric}:")
        print(f"  vs Accuracy:     r = {acc_corr:.4f}, p = {acc_p:.4f}")
        print(f"  vs Exact Match:  r = {em_corr:.4f}, p = {em_p:.4f}")
        print(f"  vs F1 Score:     r = {f1_corr:.4f}, p = {f1_p:.4f}")
    
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
            uid_metrics_key = f'uid_metrics_{i}'
            
            if uid_metrics_key in problem and metrics_key in problem:
                uid_data = problem[uid_metrics_key]
                metrics_data = problem[metrics_key]
                
                all_outputs.append({
                    'problem_id': problem_id,
                    'output_index': i,
                    'output': problem[output_key],
                    'uid_shannon_equal': uid_data.get('uid_shannon_equal', 0),
                    'accuracy': metrics_data.get('acc', 0),
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
        "uid_variance_confidence_gap", "uid_gini_confidence_gap", "uid_shannon_confidence_gap"
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
        
        level_results[level] = results
    
    return level_results


def create_level_boxplots(level_summary, level_results, outdir, filename_prefix, input_filename=None):
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
        "uid_variance_confidence_gap", "uid_gini_confidence_gap", "uid_shannon_confidence_gap"
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
        
        domain_results[domain] = results
    
    return domain_results


def create_domain_boxplots(domain_summary, domain_results, outdir, filename_prefix, input_filename=None):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--outdir", default="analysis_out")
    parser.add_argument("--analysis_by_level", default=False, required=False)
    parser.add_argument("--analysis_by_domain", default=False, required=False)

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
    filename_prefix = args.input.split("/")[-2] if "/" in args.input else args.input.replace(".json", "")
    plot_file = create_boxplots(summary, results, args.outdir, filename_prefix, args.input)
    
    # Create correlation plots
    correlation_plot_file = create_correlation_plot(data, args.outdir, filename_prefix, args.input)
    
    # Extract Shannon Equal outputs
    shannon_file = extract_shannon_equal_outputs(data, args.outdir, filename_prefix)
    
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
        
        # Create level-specific boxplots
        level_plot_file, individual_level_plots = create_level_boxplots(level_summary, level_results, args.outdir, filename_prefix, args.input)
        
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
        domain_plot_file, individual_domain_plots = create_domain_boxplots(domain_summary, domain_results, args.outdir, filename_prefix, args.input)
        
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
    print(f"Boxplots saved to: {plot_file}")
    if correlation_plot_file:
        print(f"Correlation plots saved to: {correlation_plot_file}")
    if shannon_file:
        print(f"Shannon Equal outputs saved to: {shannon_file}")


if __name__ == "__main__":
    main()