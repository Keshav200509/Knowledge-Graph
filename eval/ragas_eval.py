import os
import json
import pandas as pd
import numpy as np
import argparse
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, 
    context_precision, context_recall, answer_correctness
)

def bootstrap_metric(values, n_boot=1000):
    """Calculates 95% Confidence Interval via Bootstrapping[cite: 28]."""
    means = []
    for _ in range(n_boot):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    return np.percentile(means, 2.5), np.percentile(means, 97.5)

def run_eval(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # RAGAS expects specific column names
    dataset = Dataset.from_list(data)
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
    )
    
    df = result.to_pandas()
    summary = {}
    
    for metric in ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']:
        score = df[metric].mean()
        lower, upper = bootstrap_metric(df[metric].dropna().values)
        summary[metric] = f"{score:.3f} [{lower:.3f}, {upper:.3f}]" # Format required for table 
        
    return summary, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_results', type=str, default='results/')
    args = parser.parse_args()

    table_data = []
    for t in ['04', '06']:
        path = os.path.join(args.input_results, f'raw_{t}.json')
        print(f"📊 Evaluating Configuration t=0.{t}...")
        scores, raw_df = run_eval(path)
        scores['System'] = f"(PRM t=0.{t})"
        table_data.append(scores)
        raw_df.to_csv(f"results/metrics_t{t}.csv", index=False)

    summary_df = pd.DataFrame(table_data)
    summary_df.to_csv("results/final_results_table.csv", index=False)
    print("\n--- FINAL RESEARCH RESULTS ---")
    print(summary_df.to_string(index=False))