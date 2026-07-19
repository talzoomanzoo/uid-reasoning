import json
import argparse
from dataclasses import dataclass
import os
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

EXCLUDED_OVERALL_METRICS = {
    "total_time",
    "overall_mean_upper_bound_accuracy",
    "overall_mean_validity",
}

CONJUNCTION_ALPHA = 0.5
CONJUNCTION_BETA = 0.5


def filter_recorded_overall_metrics(overall_metrics):
    """Keep metrics that should be copied into analysis outputs."""
    if not overall_metrics:
        return {}
    return {
        key: value
        for key, value in overall_metrics.items()
        if key not in EXCLUDED_OVERALL_METRICS
    }


def format_metric_label(metric_name):
    return metric_name.replace("overall_mean_", "").replace("_", " ").title()


def format_metric_value(value):
    if isinstance(value, (int, float)):
        return f"{value:.3f}"
    return str(value)


def add_recorded_metrics_box(fig, overall_metrics):
    recorded_metrics = filter_recorded_overall_metrics(overall_metrics)
    if not recorded_metrics:
        return

    metrics_text = "Recorded overall metrics\n" + "\n".join(
        f"{format_metric_label(key)}: {format_metric_value(value)}"
        for key, value in recorded_metrics.items()
    )
    fig.text(
        0.99,
        0.02,
        metrics_text,
        ha="right",
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.85},
    )


def add_recorded_metric_lines(ax, overall_metrics):
    recorded_metrics = filter_recorded_overall_metrics(overall_metrics)
    if not recorded_metrics:
        return

    colors = ["red", "blue", "green", "purple", "orange", "brown", "gray"]
    for idx, (key, value) in enumerate(recorded_metrics.items()):
        if not isinstance(value, (int, float)):
            continue
        ax.axhline(
            y=value,
            color=colors[idx % len(colors)],
            linestyle=":",
            linewidth=2,
            alpha=0.85,
            label=f"{format_metric_label(key)} ({value:.4f})",
        )


def analyze_uid_accuracy(data):
    """
    Analyze accuracy based on UID scores.
    For each problem, find outputs with highest and lowest UID scores for each metric,
    then calculate their accuracy.
    """
    
    # Global UID is variance entropy only.
    uid_metrics = ["uid_variance_entropy"]
    
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
    """Create the global UID plot for uid_variance_entropy highest/lowest accuracy."""
    plt.style.use('default')
    sns.set_palette("husl")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    if input_filename:
        dataset_name = input_filename.split("/")[-2] if "/" in input_filename else input_filename.replace(".json", "")
        fig.suptitle('Global UID: uid_variance_entropy \n ' + dataset_name, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Global UID: uid_variance_entropy \n ' + filename_prefix, fontsize=16, fontweight='bold')

    metric = "uid_variance_entropy"
    labels = ["Highest Variance Entropy", "Lowest Variance Entropy"]
    values = [
        summary['highest_uid'][metric]['mean_accuracy'],
        summary['lowest_uid'][metric]['mean_accuracy'],
    ]

    x = np.arange(len(labels))
    bars = ax.bar(x, values, 0.6, color=['lightblue', 'lightcoral'], alpha=0.8)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f'{value:.4f}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=10,
        )

    add_recorded_metric_lines(ax, overall_metrics)
    add_recorded_metrics_box(fig, overall_metrics)

    ax.set_title('Global UID: highest/lowest uid_variance_entropy')
    ax.set_ylabel('Mean Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plot_file = os.path.join(outdir, f"{filename_prefix}_global_uid_variance_entropy.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_file


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
        return 0, 0
    
    # Calculate mean and standard deviation
    mean_diff = sum(all_diffs) / len(all_diffs)
    variance = sum((x - mean_diff) ** 2 for x in all_diffs) / len(all_diffs)
    std_diff = variance ** 0.5
    
    print(f"calculate_sigma_thresholds: Mean={mean_diff:.6f}, Std={std_diff:.6f}")
    
    # Local UID uses 3-sigma spike/fall thresholds.
    threshold_3sigma_pos = mean_diff + 3 * std_diff
    threshold_3sigma_neg = mean_diff - 3 * std_diff
    
    print(f"calculate_sigma_thresholds: 3-sigma thresholds: pos={threshold_3sigma_pos:.6f}, neg={threshold_3sigma_neg:.6f}")
    
    return threshold_3sigma_pos, threshold_3sigma_neg


def count_spikes_falls(arr, threshold_pos, threshold_neg):
    """Count spikes and falls based on the supplied local UID thresholds."""
    if not isinstance(arr, list) or len(arr) < 2:
        return {"spikes": 0, "falls": 0, "zeros": 0}
    
    spikes = falls = zeros = 0
    prev = arr[0]
    
    for x in arr[1:]:
        try:
            dx = float(x) - float(prev)
            if dx > threshold_pos:
                spikes += 1
            elif dx < threshold_neg:
                falls += 1
            else:
                zeros += 1
        except Exception:
            pass
        prev = x
    
    return {"spikes": spikes, "falls": falls, "zeros": zeros}


def analyze_spikes_falls_accuracy(data, threshold_pos, threshold_neg):
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
                    spike_fall_counts = count_spikes_falls(id_h, threshold_pos, threshold_neg)
                    
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


def analyze_conjunction_accuracy(
    data,
    threshold_pos,
    threshold_neg,
    alpha=CONJUNCTION_ALPHA,
    beta=CONJUNCTION_BETA,
):
    """
    Analyze the conjunction of local uniformity and global non-uniformity.

    U_local = 1 / (1 + alpha * spikes + beta * falls)
    G_nonuni = 1 - uid_shannon_entropy
    S_AND = U_local * G_nonuni
    """
    results = {
        "highest_conjunction": defaultdict(list),
        "lowest_conjunction": defaultdict(list),
    }

    for problem in data:
        problem_id = problem.get("id", "unknown")
        outputs = []
        i = 0

        while f"Output_{i}" in problem:
            metrics_key = f"Metrics_{i}"
            uid_metrics_key = f"id_metrics_{i}_metrics"
            id_h_key = f"id_h_{i}"

            if (
                metrics_key in problem
                and uid_metrics_key in problem
                and id_h_key in problem
            ):
                id_h = problem[id_h_key]
                uid_metrics = problem[uid_metrics_key]

                if (
                    isinstance(id_h, list)
                    and len(id_h) >= 2
                    and "uid_shannon_entropy" in uid_metrics
                ):
                    counts = count_spikes_falls(
                        id_h, threshold_pos, threshold_neg
                    )
                    g_uni = float(uid_metrics["uid_shannon_entropy"])
                    if not np.isfinite(g_uni):
                        i += 1
                        continue

                    g_uni = float(np.clip(g_uni, 0.0, 1.0))
                    u_local = 1.0 / (
                        1.0
                        + alpha * counts["spikes"]
                        + beta * counts["falls"]
                    )
                    g_nonuni = 1.0 - g_uni
                    s_and = u_local * g_nonuni
                    metrics_data = problem[metrics_key]

                    outputs.append({
                        "index": i,
                        "s_and": s_and,
                        "u_local": u_local,
                        "g_uni": g_uni,
                        "g_nonuni": g_nonuni,
                        "spikes": counts["spikes"],
                        "falls": counts["falls"],
                        "accuracy": metrics_data.get("math_equal", 0),
                        "exact_match": metrics_data.get("em", 0),
                        "f1": metrics_data.get("f1", 0),
                    })
            i += 1

        if not outputs:
            continue

        highest_output = max(outputs, key=lambda output: output["s_and"])
        lowest_output = min(outputs, key=lambda output: output["s_and"])

        for result_type, selected_output in (
            ("highest_conjunction", highest_output),
            ("lowest_conjunction", lowest_output),
        ):
            results[result_type]["s_and"].append({
                "problem_id": problem_id,
                "output_index": selected_output["index"],
                "s_and": selected_output["s_and"],
                "u_local": selected_output["u_local"],
                "g_uni": selected_output["g_uni"],
                "g_nonuni": selected_output["g_nonuni"],
                "spikes_count": selected_output["spikes"],
                "falls_count": selected_output["falls"],
                "accuracy": selected_output["accuracy"],
                "exact_match": selected_output["exact_match"],
                "f1": selected_output["f1"],
            })

    return results


def calculate_conjunction_summary_stats(results):
    """Calculate summary statistics for highest/lowest conjunction scores."""
    summary = {}

    for result_type in ("highest_conjunction", "lowest_conjunction"):
        items = results[result_type].get("s_and", [])
        summary[result_type] = {
            "s_and": {
                "count": len(items),
                "alpha": CONJUNCTION_ALPHA,
                "beta": CONJUNCTION_BETA,
                "mean_accuracy": float(np.mean([
                    float(item["accuracy"]) for item in items
                ])) if items else 0.0,
                "std_accuracy": float(np.std([
                    float(item["accuracy"]) for item in items
                ])) if items else 0.0,
                "mean_exact_match": float(np.mean([
                    item["exact_match"] for item in items
                ])) if items else 0.0,
                "std_exact_match": float(np.std([
                    item["exact_match"] for item in items
                ])) if items else 0.0,
                "mean_f1": float(np.mean([
                    item["f1"] for item in items
                ])) if items else 0.0,
                "std_f1": float(np.std([
                    item["f1"] for item in items
                ])) if items else 0.0,
                "mean_s_and": float(np.mean([
                    item["s_and"] for item in items
                ])) if items else 0.0,
                "std_s_and": float(np.std([
                    item["s_and"] for item in items
                ])) if items else 0.0,
                "mean_u_local": float(np.mean([
                    item["u_local"] for item in items
                ])) if items else 0.0,
                "mean_g_nonuni": float(np.mean([
                    item["g_nonuni"] for item in items
                ])) if items else 0.0,
                "mean_spikes_count": float(np.mean([
                    item["spikes_count"] for item in items
                ])) if items else 0.0,
                "mean_falls_count": float(np.mean([
                    item["falls_count"] for item in items
                ])) if items else 0.0,
            }
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
        fig.suptitle('Local UID: spikes_and_falls_3sigma \n ' + dataset_name, fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Local UID: spikes_and_falls_3sigma \n ' + filename_prefix, fontsize=16, fontweight='bold')
    
    # Prepare data - now using the combined structure
    analysis_types = ['highest_spikes_falls', 'lowest_spikes_falls']
    type_labels = ['Highest spikes_and_falls_3sigma', 'Lowest spikes_and_falls_3sigma']
    
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
        plot_file = os.path.join(outdir, f"{filename_prefix}_local_uid_spikes_and_falls_3sigma.png")
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

    add_recorded_metric_lines(ax, overall_metrics)
    add_recorded_metrics_box(fig, overall_metrics)

    ax.set_title('Local UID: highest/lowest spikes_and_falls_3sigma')
    ax.set_ylabel('Mean Accuracy')
    ax.set_xticks(x)
    ax.set_xticklabels(type_labels[:len(mean_accuracies)], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Save the plot
    plot_file = os.path.join(outdir, f"{filename_prefix}_local_uid_spikes_and_falls_3sigma.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file


def create_conjunction_plot(
    conjunction_summary,
    outdir,
    filename_prefix,
    input_filename=None,
    overall_metrics=None,
):
    """Plot accuracy selected by highest and lowest S_AND scores."""
    plt.style.use("default")
    sns.set_palette("husl")

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    dataset_name = (
        input_filename.split("/")[-2]
        if input_filename and "/" in input_filename
        else filename_prefix
    )
    fig.suptitle(
        f"Conjunction UID: S_AND\n{dataset_name}",
        fontsize=16,
        fontweight="bold",
    )

    result_types = ("highest_conjunction", "lowest_conjunction")
    labels = ("Highest S_AND", "Lowest S_AND")
    values = [
        conjunction_summary[result_type]["s_and"]["mean_accuracy"]
        for result_type in result_types
    ]
    x = np.arange(len(labels))
    bars = ax.bar(
        x,
        values,
        0.6,
        color=["lightblue", "lightcoral"],
        alpha=0.8,
    )

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    add_recorded_metric_lines(ax, overall_metrics)
    add_recorded_metrics_box(fig, overall_metrics)
    ax.set_title(
        "Local uniformity AND global non-uniformity "
        f"(alpha={CONJUNCTION_ALPHA}, beta={CONJUNCTION_BETA})"
    )
    ax.set_ylabel("Mean Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)

    plot_file = os.path.join(
        outdir, f"{filename_prefix}_conjunction_s_and.png"
    )
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    return plot_file


def create_combined_analysis_plot(summary, spikes_falls_summary_3sigma, outdir, filename_prefix, input_filename=None, overall_metrics=None):
    """Create combined plot for global variance entropy and local 3-sigma spike/fall UID."""
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    if input_filename:
        dataset_name = input_filename.split("/")[-1].replace(".json", "")
        fig.suptitle(f'Global UID and Local UID Analysis: {dataset_name}', fontsize=16, fontweight='bold')
    else:
        fig.suptitle('Global UID and Local UID Analysis', fontsize=16, fontweight='bold')
    
    # Left plot: global UID analysis - variance entropy only.
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

    add_recorded_metric_lines(ax1, overall_metrics)

    ax1.set_title('Global UID: highest/lowest uid_variance_entropy', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Mean Accuracy', fontsize=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels, rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Helper function to create spike/fall plots - UPDATED for combined structure
    def create_spike_fall_plot(ax, spikes_falls_summary, title, overall_metrics):
        analysis_types = ['highest_spikes_falls', 'lowest_spikes_falls']
        type_labels = ['Highest spikes_and_falls_3sigma', 'Lowest spikes_and_falls_3sigma']
        
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

        add_recorded_metric_lines(ax, overall_metrics)

        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Mean Accuracy', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels(type_labels[:len(mean_accuracies)], rotation=45, ha='right', fontsize=8)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)
    
    # Right plot: local UID analysis - 3-sigma spikes/falls only.
    create_spike_fall_plot(ax2, spikes_falls_summary_3sigma, 'Local UID: highest/lowest spikes_and_falls_3sigma', overall_metrics)
    add_recorded_metrics_box(fig, overall_metrics)
    
    plt.tight_layout()
    
    # Save the combined plot
    plot_file = os.path.join(outdir, f"{filename_prefix}_global_local_uid_analysis.png")
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_file




class UIDAccuracyAnalyzer:
    analyze = staticmethod(analyze_uid_accuracy)
    calculate_summary_stats = staticmethod(calculate_summary_stats)
    calculate_aggregated_stats = staticmethod(calculate_aggregated_stats)


class SpikesFallsAnalyzer:
    calculate_sigma_thresholds = staticmethod(calculate_sigma_thresholds)
    analyze = staticmethod(analyze_spikes_falls_accuracy)
    calculate_summary_stats = staticmethod(calculate_spikes_falls_summary_stats)


class ConjunctionAnalyzer:
    analyze = staticmethod(analyze_conjunction_accuracy)
    calculate_summary_stats = staticmethod(calculate_conjunction_summary_stats)


class AnalysisPlotter:
    create_boxplots = staticmethod(create_boxplots)
    create_spikes_falls_boxplots = staticmethod(create_spikes_falls_boxplots)
    create_conjunction_plot = staticmethod(create_conjunction_plot)
    create_combined_analysis_plot = staticmethod(create_combined_analysis_plot)


@dataclass
class AnalysisConfig:
    input: str
    input2: Optional[str] = None
    outdir: str = "analysis_out"


class UIDAnalysisRunner:
    def __init__(self, config: AnalysisConfig):
        self.config = config

    @classmethod
    def from_cli(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--input", type=str, required=True)
        parser.add_argument(
            "--input2",
            type=str,
            required=False,
            help="Path to metrics JSON file containing overall_mean_accuracy and overall_mean_self_certainty_accuracy",
        )
        parser.add_argument("--outdir", default="analysis_out")
        return cls(AnalysisConfig(**vars(parser.parse_args())))

    def run(self):
        os.makedirs(self.config.outdir, exist_ok=True)

        with open(self.config.input, "r") as f:
            data = json.load(f)

        overall_metrics = None
        if self.config.input2:
            with open(self.config.input2, "r") as f:
                metrics_data = json.load(f)
                overall_metrics = metrics_data.get("overall", metrics_data)
        recorded_overall_metrics = filter_recorded_overall_metrics(overall_metrics)

        print(f"Analyzing {len(data)} problems...")
        if overall_metrics:
            print(f"Overall metrics loaded: {overall_metrics}")
            print(f"Overall accuracy: {overall_metrics.get('overall_mean_accuracy', 'Not found')}")
            print(f"Overall self-certainty: {overall_metrics.get('overall_mean_self_certainty_accuracy', 'Not found')}")

        results = UIDAccuracyAnalyzer.analyze(data)
        summary = UIDAccuracyAnalyzer.calculate_summary_stats(results)
        aggregated = UIDAccuracyAnalyzer.calculate_aggregated_stats(summary)

        print("\n" + "="*80)
        print("ACCURACY ANALYSIS BASED ON GLOBAL UID SCORES")
        print("="*80)
        for uid_type in ['highest_uid', 'lowest_uid']:
            print(f"\n{uid_type.upper().replace('_', ' ')} UID SCORES:")
            print("-" * 60)
            for metric in sorted(summary[uid_type].keys()):
                stats = summary[uid_type][metric]
                print(f"{metric:35} | Accuracy: {stats['mean_accuracy']:.4f} +/- {stats['std_accuracy']:.4f} | "
                      f"Exact Match: {stats['mean_exact_match']:.4f} +/- {stats['std_exact_match']:.4f} | "
                      f"F1: {stats['mean_f1']:.4f} +/- {stats['std_f1']:.4f} | "
                      f"UID Score: {stats['mean_uid_score']:.4f} +/- {stats['std_uid_score']:.4f}")

        print("\n" + "="*80)
        print("AGGREGATED STATISTICS ACROSS GLOBAL UID METRICS")
        print("="*80)
        for uid_type in ['highest_uid', 'lowest_uid']:
            print(f"\n{uid_type.upper().replace('_', ' ')} UID SCORES (AGGREGATED):")
            print("-" * 60)
            stats = aggregated[uid_type]
            print(f"Mean Accuracy: {stats['mean_accuracy']:.4f} +/- {stats['std_accuracy']:.4f}")
            print(f"Mean Accuracy Std: {stats['mean_accuracy_std']:.4f} +/- {stats['std_accuracy_std']:.4f}")
            print(f"Mean Exact Match: {stats['mean_exact_match']:.4f} +/- {stats['mean_exact_match_std']:.4f}")
            print(f"Mean Exact Match Std: {stats['mean_exact_match_std']:.4f} +/- {stats['std_exact_match_std']:.4f}")
            print(f"Mean F1: {stats['mean_f1']:.4f} +/- {stats['std_f1']:.4f}")
            print(f"Mean F1 Std: {stats['mean_f1_std']:.4f} +/- {stats['std_f1_std']:.4f}")
            print(f"Mean UID Score: {stats['mean_uid_score']:.4f} +/- {stats['std_uid_score']:.4f}")
            print(f"Mean UID Score Std: {stats['mean_uid_score_std']:.4f} +/- {stats['std_uid_score_std']:.4f}")

        filename_prefix = self.config.input.split("/")[-2].replace("outputs_old", "outputs") if "/" in self.config.input else self.config.input.replace(".json", "").replace("outputs_old", "outputs")
        plot_file = AnalysisPlotter.create_boxplots(summary, results, self.config.outdir, filename_prefix, self.config.input, overall_metrics)

        print("\n" + "="*80)
        print("LOCAL UID SPIKE/FALL ANALYSIS")
        print("="*80)
        threshold_3sigma_pos, threshold_3sigma_neg = SpikesFallsAnalyzer.calculate_sigma_thresholds(data)
        print(f"3-sigma thresholds: pos={threshold_3sigma_pos:.4f}, neg={threshold_3sigma_neg:.4f}")

        spikes_falls_results_3sigma = SpikesFallsAnalyzer.analyze(data, threshold_3sigma_pos, threshold_3sigma_neg)
        spikes_falls_summary_3sigma = SpikesFallsAnalyzer.calculate_summary_stats(spikes_falls_results_3sigma)

        conjunction_results = ConjunctionAnalyzer.analyze(
            data, threshold_3sigma_pos, threshold_3sigma_neg
        )
        conjunction_summary = ConjunctionAnalyzer.calculate_summary_stats(
            conjunction_results
        )

        print("\n" + "="*80)
        print("ACCURACY ANALYSIS BASED ON 3-SIGMA SPIKE/FALL COUNTS")
        print("="*80)
        for analysis_type in ['highest_spikes_falls', 'lowest_spikes_falls']:
            print(f"\n{analysis_type.upper().replace('_', ' ')}:")
            print("-" * 60)
            for metric in spikes_falls_summary_3sigma[analysis_type]:
                stats = spikes_falls_summary_3sigma[analysis_type][metric]
                print(f"{metric:35} | Accuracy: {stats['mean_accuracy']:.4f} +/- {stats['std_accuracy']:.4f} | "
                      f"Exact Match: {stats['mean_exact_match']:.4f} +/- {stats['std_exact_match']:.4f} | "
                      f"F1: {stats['mean_f1']:.4f} +/- {stats['std_f1']:.4f} | "
                      f"Combined Count: {stats['mean_combined_count']:.2f} +/- {stats['std_combined_count']:.2f} | "
                      f"Spikes: {stats['mean_spikes_count']:.2f} +/- {stats['std_spikes_count']:.2f} | "
                      f"Falls: {stats['mean_falls_count']:.2f} +/- {stats['std_falls_count']:.2f}")

        spikes_falls_plot_file = AnalysisPlotter.create_spikes_falls_boxplots(
            spikes_falls_summary_3sigma, self.config.outdir, filename_prefix, self.config.input, overall_metrics
        )

        spikes_falls_results_file_3sigma = os.path.join(
            self.config.outdir, self.config.input.split("/")[-2] + "_spikes_and_falls_3sigma_analysis.json"
        )
        with open(spikes_falls_results_file_3sigma, "w") as f:
            json.dump(spikes_falls_results_3sigma, f, indent=2)

        spikes_falls_summary_file_3sigma = os.path.join(
            self.config.outdir, self.config.input.split("/")[-2] + "_spikes_and_falls_3sigma_summary.json"
        )
        with open(spikes_falls_summary_file_3sigma, "w") as f:
            json.dump(spikes_falls_summary_3sigma, f, indent=2)

        print("\n" + "="*80)
        print("CONJUNCTION ANALYSIS: LOCAL UNIFORMITY AND GLOBAL NON-UNIFORMITY")
        print("="*80)
        for result_type in ("highest_conjunction", "lowest_conjunction"):
            stats = conjunction_summary[result_type]["s_and"]
            print(
                f"{result_type.upper().replace('_', ' '):35} | "
                f"Accuracy: {stats['mean_accuracy']:.4f} +/- "
                f"{stats['std_accuracy']:.4f} | "
                f"S_AND: {stats['mean_s_and']:.4f} +/- "
                f"{stats['std_s_and']:.4f} | "
                f"U_local: {stats['mean_u_local']:.4f} | "
                f"G_nonuni: {stats['mean_g_nonuni']:.4f}"
            )

        conjunction_results_file = os.path.join(
            self.config.outdir,
            self.config.input.split("/")[-2] + "_conjunction_s_and_analysis.json",
        )
        with open(conjunction_results_file, "w") as f:
            json.dump(conjunction_results, f, indent=2)

        conjunction_summary_file = os.path.join(
            self.config.outdir,
            self.config.input.split("/")[-2] + "_conjunction_s_and_summary.json",
        )
        with open(conjunction_summary_file, "w") as f:
            json.dump(conjunction_summary, f, indent=2)

        conjunction_plot_file = AnalysisPlotter.create_conjunction_plot(
            conjunction_summary,
            self.config.outdir,
            filename_prefix,
            self.config.input,
            overall_metrics,
        )

        results_file = os.path.join(self.config.outdir, self.config.input.split("/")[-2] + "_uid_analysis.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        summary_file = os.path.join(self.config.outdir, self.config.input.split("/")[-2] + "_uid_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        aggregated_file = os.path.join(self.config.outdir, self.config.input.split("/")[-2] + "_uid_aggregated.json")
        with open(aggregated_file, "w") as f:
            json.dump(aggregated, f, indent=2)

        recorded_overall_metrics_file = os.path.join(
            self.config.outdir,
            self.config.input.split("/")[-2] + "_overall_metrics.json",
        )
        with open(recorded_overall_metrics_file, "w") as f:
            json.dump({"overall": recorded_overall_metrics}, f, indent=2)

        combined_plot_file = AnalysisPlotter.create_combined_analysis_plot(
            summary, spikes_falls_summary_3sigma, self.config.outdir, filename_prefix, self.config.input, overall_metrics
        )

        print(f"\nDetailed results saved to: {results_file}")
        print(f"Summary saved to: {summary_file}")
        print(f"Aggregated results saved to: {aggregated_file}")
        print(f"Overall metrics saved to: {recorded_overall_metrics_file}")
        print(f"Global UID analysis plot saved to: {plot_file}")
        print(f"Spike/Fall plot saved to: {spikes_falls_plot_file}")
        print(f"Spike/Fall analysis (3-sigma thresholds) saved to: {spikes_falls_results_file_3sigma}")
        print(f"Spike/Fall summary (3-sigma thresholds) saved to: {spikes_falls_summary_file_3sigma}")
        print(f"Conjunction analysis saved to: {conjunction_results_file}")
        print(f"Conjunction summary saved to: {conjunction_summary_file}")
        print(f"Conjunction plot saved to: {conjunction_plot_file}")
        print(f"Combined analysis plot saved to: {combined_plot_file}")

def main():
    UIDAnalysisRunner.from_cli().run()

if __name__ == "__main__":
    main()