import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


EQUAL_KEYS = ["uid_variance_equal", "uid_gini_equal", "uid_shannon_equal"]
LOGPROB_KEYS = ["uid_variance_logprob", "uid_gini_logprob", "uid_shannon_logprob"]
ENTROPY_KEYS = ["uid_variance_entropy", "uid_gini_entropy", "uid_shannon_entropy"]
CONF_GAP_KEYS = ["uid_variance_confidence_gap", "uid_gini_confidence_gap", "uid_shannon_confidence_gap"]


def load_records(path: str) -> List[Dict[str, Any]]:
    """Load a list of records from a JSON array or JSON Lines file."""
    # Try JSON array first
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try to find the longest list of dicts in top-level values
            candidates = [v for v in data.values() if isinstance(v, list) and all(isinstance(x, dict) for x in v)]
            if candidates:
                records = max(candidates, key=len)
                return records
            return [data]
    except json.JSONDecodeError:
        pass  # fall back to JSON Lines

    # Try JSON Lines (ndjson)
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                records.append(obj)
            except json.JSONDecodeError:
                continue
    if not records:
        raise ValueError("No records could be loaded; file is not valid JSON or JSON Lines.")
    return records


def iter_dicts(d: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    """Yield (key, subdict) pairs for all nested dicts within d (including top-level dict values)."""
    for k, v in d.items():
        if isinstance(v, dict):
            yield k, v
            for subk, subd in iter_dicts(v):
                yield f"{k}.{subk}", subd
        elif isinstance(v, list):
            for i, item in enumerate(v):
                if isinstance(item, dict):
                    for subk, subd in iter_dicts(item):
                        yield f"{k}[{i}].{subk}", subd


def find_uid_metrics_dict(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a dict within rec whose key starts with 'uid_metrics' and return that dict."""
    for k, v in rec.items():
        if k.startswith("uid_metrics") and isinstance(v, dict):
            return v
    for k, subd in iter_dicts(rec):
        parts = k.split(".")
        if any(part.startswith("uid_metrics") for part in parts):
            return subd
    return None


def find_metrics_dict(rec: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Find a dict within rec whose key starts with 'Metrics' and return that dict."""
    for k, v in rec.items():
        if k.startswith("Metrics") and isinstance(v, dict):
            return v
    for k, subd in iter_dicts(rec):
        parts = k.split(".")
        if any(part.startswith("Metrics") for part in parts):
            return subd
    return None


def collect_metrics(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """Collect the nine metrics and math_equal status from each record into a DataFrame (one row per record)."""
    rows = []
    for rec in records:
        md = find_uid_metrics_dict(rec)
        metrics_md = find_metrics_dict(rec)
        if not md or not metrics_md:
            continue
        row = {}
        ok = True
        for key in EQUAL_KEYS + LOGPROB_KEYS + CONF_GAP_KEYS + ENTROPY_KEYS:
            val = md.get(key, None)
            if val is None:
                ok = False
                break
            row[key] = val
        
        # Extract math_equal status
        math_equal = metrics_md.get("math_equal", None)
        if math_equal is None:
            ok = False
        
        if ok:
            row["math_equal"] = math_equal
            rows.append(row)
    if not rows:
        raise ValueError("No records contained a complete set of UID metrics and math_equal status.")
    df = pd.DataFrame(rows)
    return df


def corr_and_pca(df: pd.DataFrame, keys: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute 3×3 Pearson correlation and PCA (loadings + explained variance) for the given keys."""
    sub = df[keys].dropna()
    if sub.shape[0] < 2:
        raise ValueError(f"Not enough rows ({sub.shape[0]}) to compute correlation/PCA for keys: {keys}")

    corr_df = sub.corr(method="pearson")

    # PCA via eigen-decomposition of covariance
    X = sub.values.astype(float)
    X = X - X.mean(axis=0, keepdims=True)
    n = X.shape[0]
    if n < 2:
        raise ValueError("Need at least 2 rows after dropping NaNs for PCA.")
    cov = (X.T @ X) / (n - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    loadings_df = pd.DataFrame(eigvecs, index=keys, columns=[f"PC{i+1}" for i in range(len(keys))])

    total_var = eigvals.sum()
    explained_ratio = eigvals / total_var if total_var > 0 else np.zeros_like(eigvals)
    explained_df = pd.DataFrame(
        {"component": [f"PC{i+1}" for i in range(len(keys))],
         "eigenvalue": eigvals,
         "explained_variance_ratio": explained_ratio}
    )

    return corr_df, loadings_df, explained_df


def annotate_heatmap(ax, mat: np.ndarray, xlabels: List[str], ylabels: List[str]) -> None:
    """Annotate a heatmap with numeric values."""
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center")


def plot_correlation(corr_df: pd.DataFrame, title: str, outpath: Optional[str], show: bool, dpi: int) -> None:
    fig = plt.figure()
    ax = fig.gca()
    im = ax.imshow(corr_df.values, aspect="equal")
    ax.set_xticks(range(corr_df.shape[1]))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(corr_df.shape[0]))
    ax.set_yticklabels(corr_df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    annotate_heatmap(ax, corr_df.values, list(corr_df.columns), list(corr_df.index))
    if outpath:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_loadings(loadings_df: pd.DataFrame, title: str, outpath: Optional[str], show: bool, dpi: int) -> None:
    fig = plt.figure()
    ax = fig.gca()
    im = ax.imshow(loadings_df.values, aspect="equal", vmin=-1, vmax=1)
    ax.set_xticks(range(loadings_df.shape[1]))
    ax.set_xticklabels(loadings_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(loadings_df.shape[0]))
    ax.set_yticklabels(loadings_df.index)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    annotate_heatmap(ax, loadings_df.values, list(loadings_df.columns), list(loadings_df.index))
    if outpath:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_explained(explained_df: pd.DataFrame, title: str, outpath: Optional[str], show: bool, dpi: int) -> None:
    fig = plt.figure()
    ax = fig.gca()
    x = np.arange(len(explained_df))
    ax.bar(x, explained_df["explained_variance_ratio"].values)
    ax.set_xticks(x)
    ax.set_xticklabels(explained_df["component"].tolist())
    ax.set_ylim(0, 1)
    ax.set_ylabel("Explained Variance Ratio")
    ax.set_title(title)
    for i, v in enumerate(explained_df["explained_variance_ratio"].values):
        ax.text(i, v + 0.02 if v < 0.95 else 0.98, f"{v:.2f}", ha="center", va="bottom")
    if outpath:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def plot_metric_comparison(df_true: pd.DataFrame, df_false: pd.DataFrame, keys: List[str], 
                          title: str, outpath: Optional[str], show: bool, dpi: int) -> None:
    """Create box plots comparing metrics between math_equal true and false groups."""
    fig, axes = plt.subplots(1, len(keys), figsize=(4*len(keys), 6))
    if len(keys) == 1:
        axes = [axes]
    
    for i, key in enumerate(keys):
        ax = axes[i]
        
        # Prepare data for box plot
        true_data = df_true[key].dropna()
        false_data = df_false[key].dropna()
        
        if len(true_data) > 0 and len(false_data) > 0:
            data = [true_data, false_data]
            labels = ['Math Equal True', 'Math Equal False']
            colors = ['lightblue', 'lightcoral']
            
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
            
            # Add mean lines
            for j, (data_group, color) in enumerate(zip(data, colors)):
                mean_val = data_group.mean()
                ax.axhline(y=mean_val, color=color, linestyle='--', alpha=0.7, 
                          label=f'Mean: {mean_val:.3f}')
            
            ax.set_title(f'{key.replace("uid_", "").replace("_", " ").title()}')
            ax.set_ylabel('Value')
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center', 
                   transform=ax.transAxes)
            ax.set_title(f'{key.replace("uid_", "").replace("_", " ").title()}')
    
    plt.suptitle(title)
    plt.tight_layout()
    
    if outpath:
        fig.savefig(outpath, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def compute_statistical_comparison(df_true: pd.DataFrame, df_false: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Compute statistical comparison between math_equal true and false groups."""
    results = []
    
    for key in keys:
        true_data = df_true[key].dropna()
        false_data = df_false[key].dropna()
        
        if len(true_data) > 0 and len(false_data) > 0:
            # Basic statistics
            true_mean = true_data.mean()
            false_mean = false_data.mean()
            true_std = true_data.std()
            false_std = false_data.std()
            
            # Mann-Whitney U test (non-parametric)
            try:
                stat, p_value = stats.mannwhitneyu(true_data, false_data, alternative='two-sided')
            except:
                stat, p_value = np.nan, np.nan
            
            # Effect size (Cohen's d)
            try:
                pooled_std = np.sqrt(((len(true_data) - 1) * true_std**2 + 
                                    (len(false_data) - 1) * false_std**2) / 
                                   (len(true_data) + len(false_data) - 2))
                cohens_d = (true_mean - false_mean) / pooled_std if pooled_std > 0 else np.nan
            except:
                cohens_d = np.nan
            
            results.append({
                'metric': key,
                'true_mean': true_mean,
                'false_mean': false_mean,
                'true_std': true_std,
                'false_std': false_std,
                'mean_diff': true_mean - false_mean,
                'mann_whitney_stat': stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'true_count': len(true_data),
                'false_count': len(false_data)
            })
        else:
            results.append({
                'metric': key,
                'true_mean': np.nan,
                'false_mean': np.nan,
                'true_std': np.nan,
                'false_std': np.nan,
                'mean_diff': np.nan,
                'mann_whitney_stat': np.nan,
                'p_value': np.nan,
                'cohens_d': np.nan,
                'true_count': len(true_data),
                'false_count': len(false_data)
            })
    
    return pd.DataFrame(results)


def main():
    parser = argparse.ArgumentParser(description="Matplotlib-only correlation and PCA plots for UID metrics.")
    parser.add_argument("--input", "-i", required=True, help="Path to JSON file (array or JSON Lines).")
    parser.add_argument("--outdir", "-o", default="uid_metrics_plots_out", help="Directory to save plots.")
    parser.add_argument("--savefigs", action="store_true", help="Save figures as PNGs in --outdir.")
    parser.add_argument("--no-show", action="store_true", help="Do not display plots (use with --savefigs).")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures.")
    parser.add_argument("--separate-analysis", action="store_true", 
                       help="Generate separate analysis for math_equal true/false groups.")
    args = parser.parse_args()

    records = load_records(args.input)
    df = collect_metrics(records)
    
    # Print summary statistics
    print(f"Total records: {len(df)}")
    print(f"Math equal True: {df['math_equal'].sum()}")
    print(f"Math equal False: {(~df['math_equal']).sum()}")
    
    # Separate data by math_equal status
    df_true = df[df['math_equal'] == True].copy()
    df_false = df[df['math_equal'] == False].copy()
    
    print(f"Records with math_equal=True: {len(df_true)}")
    print(f"Records with math_equal=False: {len(df_false)}")

    groups = [
        ("equal", EQUAL_KEYS),
        ("logprob", LOGPROB_KEYS),
        ("confidence_gap", CONF_GAP_KEYS),
        ("entropy", ENTROPY_KEYS),
    ]

    if args.savefigs:
        os.makedirs(args.outdir, exist_ok=True)

    # Generate overall analysis (original functionality)
    print("\n=== Overall Analysis (All Records) ===")
    for tag, keys in groups:
        try:
            corr_df, loadings_df, explained_df = corr_and_pca(df, keys)

            corr_path = os.path.join(args.outdir, f"corr_{tag}_overall.png") if args.savefigs else None
            loadings_path = os.path.join(args.outdir, f"pca_loadings_{tag}_overall.png") if args.savefigs else None
            explained_path = os.path.join(args.outdir, f"scree_{tag}_overall.png") if args.savefigs else None

            plot_correlation(
                corr_df,
                title=f"Correlation Matrix: {tag.replace('_', ' ').title()} (Overall)",
                outpath=corr_path,
                show=not args.no_show,
                dpi=args.dpi,
            )
            plot_loadings(
                loadings_df,
                title=f"PCA Loadings: {tag.replace('_', ' ').title()} (Overall)",
                outpath=loadings_path,
                show=not args.no_show,
                dpi=args.dpi,
            )
            plot_explained(
                explained_df,
                title=f"PCA Scree: {tag.replace('_', ' ').title()} (Overall)",
                outpath=explained_path,
                show=not args.no_show,
                dpi=args.dpi,
            )
        except Exception as e:
            print(f"Error in overall analysis for {tag}: {e}")

    # Generate separate analysis for math_equal groups
    if args.separate_analysis:
        print("\n=== Separate Analysis by Math Equal Status ===")
        
        # Analysis for math_equal=True
        if len(df_true) > 0:
            print(f"\n--- Math Equal True (n={len(df_true)}) ---")
            for tag, keys in groups:
                try:
                    corr_df, loadings_df, explained_df = corr_and_pca(df_true, keys)

                    corr_path = os.path.join(args.outdir, f"corr_{tag}_true.png") if args.savefigs else None
                    loadings_path = os.path.join(args.outdir, f"pca_loadings_{tag}_true.png") if args.savefigs else None
                    explained_path = os.path.join(args.outdir, f"scree_{tag}_true.png") if args.savefigs else None

                    plot_correlation(
                        corr_df,
                        title=f"Correlation Matrix: {tag.replace('_', ' ').title()} (Math Equal=True)",
                        outpath=corr_path,
                        show=not args.no_show,
                        dpi=args.dpi,
                    )
                    plot_loadings(
                        loadings_df,
                        title=f"PCA Loadings: {tag.replace('_', ' ').title()} (Math Equal=True)",
                        outpath=loadings_path,
                        show=not args.no_show,
                        dpi=args.dpi,
                    )
                    plot_explained(
                        explained_df,
                        title=f"PCA Scree: {tag.replace('_', ' ').title()} (Math Equal=True)",
                        outpath=explained_path,
                        show=not args.no_show,
                        dpi=args.dpi,
                    )
                except Exception as e:
                    print(f"Error in math_equal=True analysis for {tag}: {e}")
        else:
            print("No records with math_equal=True for separate analysis")

        # Analysis for math_equal=False
        if len(df_false) > 0:
            print(f"\n--- Math Equal False (n={len(df_false)}) ---")
            for tag, keys in groups:
                try:
                    corr_df, loadings_df, explained_df = corr_and_pca(df_false, keys)

                    corr_path = os.path.join(args.outdir, f"corr_{tag}_false.png") if args.savefigs else None
                    loadings_path = os.path.join(args.outdir, f"pca_loadings_{tag}_false.png") if args.savefigs else None
                    explained_path = os.path.join(args.outdir, f"scree_{tag}_false.png") if args.savefigs else None

                    plot_correlation(
                        corr_df,
                        title=f"Correlation Matrix: {tag.replace('_', ' ').title()} (Math Equal=False)",
                        outpath=corr_path,
                        show=not args.no_show,
                        dpi=args.dpi,
                    )
                    plot_loadings(
                        loadings_df,
                        title=f"PCA Loadings: {tag.replace('_', ' ').title()} (Math Equal=False)",
                        outpath=loadings_path,
                        show=not args.no_show,
                        dpi=args.dpi,
                    )
                    plot_explained(
                        explained_df,
                        title=f"PCA Scree: {tag.replace('_', ' ').title()} (Math Equal=False)",
                        outpath=explained_path,
                        show=not args.no_show,
                        dpi=args.dpi,
                    )
                except Exception as e:
                    print(f"Error in math_equal=False analysis for {tag}: {e}")
        else:
            print("No records with math_equal=False for separate analysis")

        # Generate comparison plots and statistics
        if len(df_true) > 0 and len(df_false) > 0:
            print(f"\n--- Comparison Analysis ---")
            
            # Save statistical comparison results
            all_keys = EQUAL_KEYS + LOGPROB_KEYS + CONF_GAP_KEYS + ENTROPY_KEYS
            stats_df = compute_statistical_comparison(df_true, df_false, all_keys)
            stats_path = os.path.join(args.outdir, "statistical_comparison.csv") if args.savefigs else None
            if stats_path:
                stats_df.to_csv(stats_path, index=False)
                print(f"Statistical comparison saved to: {stats_path}")
            
            # Print summary of significant differences
            significant = stats_df[stats_df['p_value'] < 0.05]
            if len(significant) > 0:
                print(f"\nSignificant differences (p < 0.05):")
                for _, row in significant.iterrows():
                    print(f"  {row['metric']}: p={row['p_value']:.4f}, Cohen's d={row['cohens_d']:.3f}")
            else:
                print("\nNo significant differences found (p < 0.05)")

            # Generate comparison plots for each group
            for tag, keys in groups:
                try:
                    comparison_path = os.path.join(args.outdir, f"comparison_{tag}.png") if args.savefigs else None
                    plot_metric_comparison(
                        df_true, df_false, keys,
                        title=f"Metric Comparison: {tag.replace('_', ' ').title()}",
                        outpath=comparison_path,
                        show=not args.no_show,
                        dpi=args.dpi,
                    )
                except Exception as e:
                    print(f"Error in comparison analysis for {tag}: {e}")

    print("Done.")


if __name__ == "__main__":
    main()
