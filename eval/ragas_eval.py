import os
import json
import pandas as pd
import numpy as np
import argparse
from ragas import evaluate
from ragas.metrics import (
    faithfulness, answer_relevancy, 
    context_precision, context_recall, answer_correctness
)
from datasets import Dataset

def bootstrap_ci(data, n_iterations=1000, alpha=0.05):
    """Calculates 95% Bootstrap Confidence Intervals."""
    stats = []
    for _ in range(n_iterations):
        resample = np.random.choice(data, size=len(data), replace=True)
        stats.append(np.mean(resample))
    lower = np.percentile(stats, (alpha / 2) * 100)
    upper = np.percentile(stats, (1 - alpha / 2) * 100)
    return lower, upper

def run_evaluation(input_file):
    # Load your raw pipeline output
    with open(input_file, 'r') as f:
        data = json.load(f) # Expected: List of {question, answer, contexts, ground_truth}
    
    dataset = Dataset.from_list(data)
    
    # Execute RAGAS evaluation
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness]
    )
    
    df = result.to_pandas()
    
    # Calculate Mean and CIs for each metric
    summary = {}
    metrics_list = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall', 'answer_correctness']
    
    for m in metrics_list:
        mean_val = df[m].mean()
        lower, upper = bootstrap_ci(df[m].values)
        summary[m] = f"{mean_val:.3f} [{lower:.3f}, {upper:.3f}]"
        
    return summary, df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_results', type=str, default='results/')
    args = parser.parse_args()

    final_table = []
    
    # Run for both ablations as required
    for t in ['04', '06']:
        input_path = os.path.join(args.input_results, f'raw_{t}.json')
        print(f"Evaluating threshold t=0.{t}...")
        scores, raw_df = run_evaluation(input_path)
        scores['System'] = f"(PRM t=0.{t})"
        final_table.append(scores)
        
        # Save individual raw results
        raw_df.to_csv(f"results/results_t{t}.csv", index=False)

    # Final Output Formatting
    summary_df = pd.DataFrame(final_table)
    summary_df.to_csv("results/final_results_table.csv", index=False)
    summary_df.to_json("results/final_results_table.json", orient='records')
    
    print("\n--- FINAL RESULTS TABLE ---")
    print(summary_df.to_string(index=False))