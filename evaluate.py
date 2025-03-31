import numpy as np

overall_mean_entropy_spike = np.mean([item[f"output_entropy_spike_{idx}"]["entropy_diff"] for item in filtered_data])
overall_mean_entropy_drop = np.mean([item[f"output_entropy_drop_{idx}"]["entropy_diff"] for item in filtered_data]) 