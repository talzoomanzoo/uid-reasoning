#!/usr/bin/env python3
import argparse
import json
import math
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

# -------------------------
# Loading & expansion utils
# -------------------------

def load_records(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        start = f.read(2048)
        if start.lstrip().startswith("["):
            f.seek(0)
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError("JSON file is not a list of objects.")
            return data
        else:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]

def expand_to_traces(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Expand records into one row per trace. We look for nested dicts whose keys start with 'uid_metrics'.
    If multiple (e.g., uid_metrics_0, uid_metrics_1, ...), we create multiple rows.
    Also attach per-trace fields like Pred_Answer_k and Metrics_k when present.
    """
    rows = []
    for rec in records:
        base = {k: v for k, v in rec.items() if not isinstance(v, dict)}
        # collect sub-dicts; map index -> dict of features
        submaps: Dict[str, Dict[str, Any]] = {}
        for k, v in rec.items():
            if isinstance(v, dict) and k.startswith("uid_metrics"):
                idx = k[len("uid_metrics"):].strip("_")
                submaps.setdefault(idx, {})
                # flatten uid metrics
                for mk, mv in v.items():
                    submaps[idx][mk] = mv

        # Attach per-trace predictions/metrics matching index
        for k, v in rec.items():
            if isinstance(v, dict) and k.startswith("Metrics"):
                idx = k[len("Metrics"):].strip("_")
                submaps.setdefault(idx, {})
                for mk, mv in v.items():
                    submaps[idx][f"Metrics__{mk}"] = mv
            elif isinstance(v, str) and k.startswith("Pred_Answer_"):
                idx = k[len("Pred_Answer_"):]
                submaps.setdefault(idx, {})
                submaps[idx]["Pred_Answer"] = v
            elif (isinstance(v, (int, float, str))) and re.match(r"^output_tokens_\d+$", k):
                idx = k.split("_")[-1]
                submaps.setdefault(idx, {})
                submaps[idx]["output_tokens"] = v

        if not submaps:
            rows.append(base)
            continue

        for idx, feat in submaps.items():
            row = dict(base)
            row["_trace_idx"] = idx if idx != "" else "0"
            row.update(feat)
            rows.append(row)
    return pd.DataFrame(rows)

# -------------------------
# Labels & grouping
# -------------------------

def infer_label(row: pd.Series) -> int:
    """
    Infer correctness label (0/1). Priority:
      1) Metrics__acc or Metrics__is_valid_answer or Metrics__em (coerce to int/bool)
      2) 'acc' or 'is_valid_answer' in flattened columns if present
      3) Compare answer vs Pred_Answer (string or numeric equality)
    """
    for key in ["Metrics__acc", "Metrics__is_valid_answer", "Metrics__em", "Metrics__math_equal"]:
        if key in row and pd.notna(row[key]):
            val = row[key]
            if isinstance(val, bool):
                return int(val)
            try: return int(val)
            except: pass
    for key in ["acc", "is_valid_answer", "em", "math_equal"]:
        if key in row and pd.notna(row[key]):
            val = row[key]
            if isinstance(val, bool):
                return int(val)
            try: return int(val)
            except: pass
    if "answer" in row and "Pred_Answer" in row and isinstance(row["Pred_Answer"], str):
        a = str(row["answer"]).strip()
        p = str(row["Pred_Answer"]).strip()
        if a == p: return 1
        try:
            if float(a) == float(p): return 1
        except: pass
    return 0

def make_item_id(row: pd.Series) -> str:
    if "id" in row and pd.notna(row["id"]):
        return str(row["id"])
    year = str(row.get("year", ""))
    num = str(row.get("problem_number", ""))
    return f"{year}-{num}"

# -------------------------
# UID columns
# -------------------------

UID_METRIC_KEYS = {
    "shannon": ("uid_shannon_logprob", "uid_shannon_entropy", "uid_shannon_confidence_gap", "uid_shannon_equal"),
    "variance": ("uid_variance_logprob", "uid_variance_entropy", "uid_variance_confidence_gap", "uid_variance_equal"),
    "gini": ("uid_gini_logprob", "uid_gini_entropy", "uid_gini_confidence_gap", "uid_gini_equal"),
}

def extract_uid_columns(df: pd.DataFrame, uid_metric: str):
    if uid_metric not in UID_METRIC_KEYS:
        raise ValueError(f"uid_metric must be one of {list(UID_METRIC_KEYS.keys())}")
    lp_col, h_col, d_col, comp_col = UID_METRIC_KEYS[uid_metric]
    for col in [lp_col, h_col, d_col, comp_col]:
        if col not in df.columns:
            raise KeyError(f"Required column missing in data: {col}")
    return df[lp_col], df[h_col], df[d_col], df[comp_col]

# -------------------------
# CV
# -------------------------

def make_loio_splits(item_ids: np.ndarray):
    unique_items = np.unique(item_ids)
    splits = []
    for item in unique_items:
        test_idx = np.where(item_ids == item)[0]
        train_idx = np.where(item_ids != item)[0]
        if len(test_idx) == 0 or len(train_idx) == 0: continue
        splits.append((train_idx, test_idx))
    return splits

def make_item_kfold_splits(item_ids: np.ndarray, n_splits: int = 5, seed: int = 42):
    rng = np.random.default_rng(seed)
    unique_items = np.array(sorted(np.unique(item_ids)))
    rng.shuffle(unique_items)
    folds = np.array_split(unique_items, n_splits)
    splits = []
    for k in range(n_splits):
        test_items = folds[k]
        test_mask = np.isin(item_ids, test_items)
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]
        if len(test_idx) == 0 or len(train_idx) == 0: continue
        splits.append((train_idx, test_idx))
    return splits

# -------------------------
# Modeling & metrics
# -------------------------

def standardize_fit_predict(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, test_idx: np.ndarray):
    scaler = StandardScaler(with_mean=True, with_std=True)
    X_tr = scaler.fit_transform(X[train_idx])
    X_te = scaler.transform(X[test_idx])
    clf = LogisticRegression(max_iter=2000, solver="lbfgs")
    clf.fit(X_tr, y[train_idx])
    return clf.predict_proba(X_te)[:, 1]

def per_item_macro_auc(y_true: np.ndarray, y_pred: np.ndarray, item_ids: np.ndarray):
    aucs = {}
    for item in np.unique(item_ids):
        mask = (item_ids == item)
        yt = y_true[mask]; yp = y_pred[mask]
        if yt.min() == yt.max():  # AUC undefined if single class
            continue
        aucs[item] = roc_auc_score(yt, yp)
    macro_auc = float(np.mean(list(aucs.values()))) if aucs else float("nan")
    return macro_auc, aucs

def per_item_logloss(y_true: np.ndarray, y_pred: np.ndarray, item_ids: np.ndarray):
    losses = {}
    eps = 1e-12
    for item in np.unique(item_ids):
        mask = (item_ids == item)
        yt = y_true[mask]; yp = np.clip(y_pred[mask], eps, 1-eps)
        losses[item] = float(log_loss(yt, yp, labels=[0,1]))
    return losses

# -------------------------
# Bootstrap comparisons
# -------------------------

def bootstrap_item_diff(metric_by_item_a, metric_by_item_b, B=5000, seed=123):
    """
    Bootstrap difference in item-averaged metrics: mean(A) - mean(B).
    Returns (point_estimate, (ci_low, ci_high), one_sided_p that diff <= 0).
    """
    rng = np.random.default_rng(seed)
    common_items = sorted(set(metric_by_item_a.keys()) & set(metric_by_item_b.keys()))
    if not common_items:
        return float("nan"), (float("nan"), float("nan")), float("nan")
    a_vals = np.array([metric_by_item_a[i] for i in common_items])
    b_vals = np.array([metric_by_item_b[i] for i in common_items])
    point = float(np.mean(a_vals) - np.mean(b_vals))
    n = len(common_items)
    boots = []
    for _ in range(B):
        idx = rng.integers(0, n, size=n)
        boots.append(float(np.mean(a_vals[idx]) - np.mean(b_vals[idx])))
    lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    p_one_sided = float((1.0 + np.sum(np.array(boots) <= 0.0)) / (1.0 + B))
    return point, (lo, hi), p_one_sided

# -------------------------
# Main
# -------------------------

def run(input_path: str, uid_metric: str, cv: str, n_splits: int, B: int, seed: int):
    records = load_records(input_path)
    df = expand_to_traces(records)

    # Labels and groups
    df["_item"] = df.apply(make_item_id, axis=1)
    df["_y"] = df.apply(infer_label, axis=1).astype(int)

    # Controls if present (optional)
    control_cols = [c for c in ["len_tokens", "num_steps", "avg_logprob", "temperature"] if c in df.columns]

    # Extract UIDs for chosen metric
    # LP-only, H-only, D-only, and Composite (already provided as *_equal)
    lp, h, d, comp = extract_uid_columns(df, uid_metric)
    uid_frames = {"LP": lp, "H": h, "D": d, "Comp": comp}

    item_ids = df["_item"].astype(str).values
    y = df["_y"].values

    # CV splits
    if cv == "loio":
        splits = make_loio_splits(item_ids)
    else:
        splits = make_item_kfold_splits(item_ids, n_splits=n_splits, seed=seed)
    if not splits:
        raise RuntimeError("No CV splits constructed; need at least 2 distinct items.")

    results = {}
    per_item = {}

    for name, uid_series in uid_frames.items():
        X_uid = uid_series.values.reshape(-1, 1)
        if control_cols:
            X_controls = df[control_cols].values
            X = np.concatenate([X_uid, X_controls], axis=1)
        else:
            X = X_uid

        y_true_all, y_pred_all, item_all = [], [], []

        for train_idx, test_idx in splits:
            proba = standardize_fit_predict(X, y, train_idx, test_idx)
            y_true_all.append(y[test_idx])
            y_pred_all.append(proba)
            item_all.append(item_ids[test_idx])

        y_true = np.concatenate(y_true_all)
        y_pred = np.concatenate(y_pred_all)
        item_cv = np.concatenate(item_all)

        # Metrics
        macro_auc, auc_by_item = per_item_macro_auc(y_true, y_pred, item_cv)
        ll_by_item = per_item_logloss(y_true, y_pred, item_cv)
        eps = 1e-12
        ll_overall = float(log_loss(y_true, np.clip(y_pred, eps, 1-eps), labels=[0,1]))

        results[name] = {
            "macro_auc": float(macro_auc),
            "log_loss_overall": ll_overall,
            "n_items_for_auc": int(len(auc_by_item)),
            "n_items_total": int(len(np.unique(item_cv))),
            "controls_used": control_cols,
        }
        per_item[name] = {"auc_by_item": auc_by_item, "ll_by_item": ll_by_item}

    # Pairwise comparisons: Composite vs Singles
    comp_auc = per_item["Comp"]["auc_by_item"]
    comp_ll  = per_item["Comp"]["ll_by_item"]
    comparisons = {}

    for single in ["LP", "H", "D"]:
        single_auc = per_item[single]["auc_by_item"]
        single_ll  = per_item[single]["ll_by_item"]

        auc_point, auc_ci, auc_p = bootstrap_item_diff(comp_auc, single_auc, B=B, seed=seed)

        common_items_ll = sorted(set(comp_ll.keys()) & set(single_ll.keys()))
        a_vals = np.array([single_ll[i] for i in common_items_ll])
        b_vals = np.array([comp_ll[i] for i in common_items_ll])
        n = len(common_items_ll)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(B):
            idx = rng.integers(0, n, size=n)
            boots.append(float(np.mean(a_vals[idx] - b_vals[idx])))
        lo, hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        ll_point = float(np.mean(a_vals - b_vals))
        ll_p = float((1.0 + np.sum(np.array(boots) <= 0.0)) / (1.0 + B))

        comparisons[single] = {
            "auc_diff_comp_minus_single": auc_point,
            "auc_diff_ci95": [auc_ci[0], auc_ci[1]],
            "auc_one_sided_p_comp_le_single": auc_p,
            "ll_diff_single_minus_comp": ll_point,
            "ll_diff_ci95": [lo, hi],
            "ll_one_sided_p_single_le_comp": ll_p,
        }

    report = {
        "settings": {
            "input": input_path,
            "uid_metric": uid_metric,
            "cv": cv,
            "n_splits": n_splits,
            "bootstrap_B": B,
            "seed": seed,
        },
        "summary_by_uid": results,
        "comparisons_comp_vs_single": comparisons,
    }

    out_json = "/mnt/data/uid_ablation_report.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nSaved JSON report to: {out_json}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="UID ablation analysis (Composite vs LP/H/D).")
    parser.add_argument("--input", required=True, help="Path to outputs.jsonl or outputs.json")
    parser.add_argument("--uid_metric", choices=["shannon","variance","gini"], default="shannon",
                        help="Which UID operationalization to use across LP/H/D/Composite")
    parser.add_argument("--cv", choices=["loio","kfold"], default="loio",
                        help="Cross-validation scheme: leave-one-item-out or item-stratified K-fold")
    parser.add_argument("--n_splits", type=int, default=5, help="K for item K-fold CV")
    parser.add_argument("--bootstrap", type=int, default=5000, help="Item bootstrap iterations for CI")
    parser.add_argument("--seed", type=int, default=123, help="Random seed")
    args = parser.parse_args()
    run(args.input, args.uid_metric, args.cv, args.n_splits, args.bootstrap, args.seed)

if __name__ == "__main__":
    main()
