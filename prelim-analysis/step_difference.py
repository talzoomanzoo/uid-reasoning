#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime
from collections import defaultdict

def count_spikes_falls(arr):
    if not isinstance(arr, list) or len(arr) < 2:
        return {"spikes": 0, "falls": 0, "zeros": 0}
    spikes = falls = zeros = 0
    prev = arr[0]
    for x in arr[1:]:
        try:
            dx = float(x) - float(prev)
            if dx > 0:
                spikes += 1
            elif dx < 0:
                falls += 1
            else:
                zeros += 1
        except Exception:
            pass
        prev = x
    return {"spikes": spikes, "falls": falls, "zeros": zeros}

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 step_difference.py /absolute/path/to/input.json")
        sys.exit(1)

    in_path = sys.argv[1]
    with open(in_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    per_output_rows = []
    # grouped by math_equal -> collect spike/fall counts to compute averages
    grouped_counts = {
        "True": {"spikes": [], "falls": []},
        "False": {"spikes": [], "falls": []},
    }

    # Expect a list of items; each may have multiple outputs: id_h_0, Metrics_0, etc.
    for item in data:
        qid = item.get("id")

        i = 0
        while True:
            h_key = f"id_h_{i}"
            m_key = f"Metrics_{i}"
            if h_key not in item:
                break

            id_h = item.get(h_key, [])
            metrics = item.get(m_key, {}) if isinstance(item.get(m_key), dict) else {}
            math_equal = bool(metrics.get("math_equal", False))
            me_key = "True" if math_equal else "False"

            counts = count_spikes_falls(id_h)
            grouped_counts[me_key]["spikes"].append(counts["spikes"])
            grouped_counts[me_key]["falls"].append(counts["falls"])

            per_output_rows.append({
                "id": qid,
                "output_index": i,
                "spikes_count": counts["spikes"],
                "falls_count": counts["falls"],
                "zeros_count": counts["zeros"],
                "math_equal": math_equal,
                "length_id_h": len(id_h)
            })

            i += 1

    def avg(lst):
        return (sum(lst) / len(lst)) if lst else 0.0

    summary = {
        "metadata": {
            "source_file": in_path,
            "num_questions": len(data) if isinstance(data, list) else 1,
            "generated_at": datetime.now().isoformat(),
        },
        "per_output_counts": per_output_rows,  # per question/output with math_equal
        "averages_by_math_equal": {
            "True": {
                "avg_spikes": avg(grouped_counts["True"]["spikes"]),
                "avg_falls": avg(grouped_counts["True"]["falls"]),
                "num_outputs": len(grouped_counts["True"]["spikes"]),
            },
            "False": {
                "avg_spikes": avg(grouped_counts["False"]["spikes"]),
                "avg_falls": avg(grouped_counts["False"]["falls"]),
                "num_outputs": len(grouped_counts["False"]["spikes"]),
            },
        },
        "totals_by_math_equal": {
            "True": {
                "total_spikes": sum(grouped_counts["True"]["spikes"]),
                "total_falls": sum(grouped_counts["True"]["falls"]),
                "num_outputs": len(grouped_counts["True"]["spikes"]),
            },
            "False": {
                "total_spikes": sum(grouped_counts["False"]["spikes"]),
                "total_falls": sum(grouped_counts["False"]["falls"]),
                "num_outputs": len(grouped_counts["False"]["spikes"]),
            },
        }
    }

    out_dir = os.path.dirname(in_path)
    out_name = f"step_difference_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path = os.path.join(out_dir, out_name)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved summary to: {out_path}")

if __name__ == "__main__":
    main()

