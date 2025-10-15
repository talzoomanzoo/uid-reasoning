#!/usr/bin/env python3
import os
import json
from glob import glob
from collections import defaultdict
from datetime import datetime

BASE_DIR = "/workspace/uid-reasoning-prelim/scripts/outputs/runs.baselines/outlier_tokens_aime.test.Qwen"

def safe_avg(total, count):
    return (total / count) if count else 0.0

def main():
    files = sorted(glob(os.path.join(BASE_DIR, "*.json")))
    by_sample = defaultdict(lambda: {
        "True": 0,   # total num_outliers where math_equal==True
        "False": 0,  # total num_outliers where math_equal==False
        "counts_by_math_equal": {"True": 0, "False": 0}  # number of files per math_equal
    })
    per_file = []  # optional detail per file

    # Global counters
    global_total = {"True": 0, "False": 0}
    global_counts = {"True": 0, "False": 0}

    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        meta = data.get("metadata", {})
        sid = meta.get("sample_id")
        math_equal = meta.get("math_equal")
        num_outliers = meta.get("num_outliers")

        if sid is None or math_equal is None or num_outliers is None:
            continue

        me_key = "True" if bool(math_equal) else "False"
        try:
            n = int(num_outliers)
        except Exception:
            continue

        # per-sample totals/counts
        by_sample[sid][me_key] += n
        by_sample[sid]["counts_by_math_equal"][me_key] += 1

        # global totals/counts
        global_total[me_key] += n
        global_counts[me_key] += 1

        per_file.append({
            "file": os.path.basename(fp),
            "sample_id": sid,
            "math_equal": bool(math_equal),
            "num_outliers": n,
        })

    # Per-sample averages
    by_sample_avg = {}
    for sid, agg in by_sample.items():
        ct_true = agg["counts_by_math_equal"]["True"]
        ct_false = agg["counts_by_math_equal"]["False"]
        by_sample_avg[str(sid)] = {
            "avg_outliers": {
                "True": safe_avg(agg["True"], ct_true),
                "False": safe_avg(agg["False"], ct_false),
            },
            "totals": {
                "True": agg["True"],
                "False": agg["False"],
            },
            "counts_by_math_equal": agg["counts_by_math_equal"],
        }

    # Overall averages (global, per file) by math_equal
    overall_avg_outliers_per_file = {
        "True": safe_avg(global_total["True"], global_counts["True"]),
        "False": safe_avg(global_total["False"], global_counts["False"]),
    }

    summary = {
        "metadata": {
            "source_dir": BASE_DIR,
            "num_files": len(files),
            "generated_at": datetime.now().isoformat(),
        },
        "by_sample": {str(k): {
            "totals": {"True": v["True"], "False": v["False"]},
            "counts_by_math_equal": v["counts_by_math_equal"],
        } for k, v in sorted(by_sample.items(), key=lambda x: x[0])},
        "by_sample_avg_outliers": by_sample_avg,
        "totals": {
            "num_outliers": {"True": global_total["True"], "False": global_total["False"]},
            "counts_by_math_equal": {"True": global_counts["True"], "False": global_counts["False"]},
        },
        "overall_avg_outliers_per_file": overall_avg_outliers_per_file,
        "details_per_file": per_file,  # optional
    }

    out_path = os.path.join(
        BASE_DIR, f"outlier_counts_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {out_path}")

if __name__ == "__main__":
    main()

