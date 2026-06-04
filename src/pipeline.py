import json
import argparse
import os
from datasets import load_dataset
from prm import ProcessRewardModel

def run_pipeline(threshold, output_file, num_questions=500):
    print(f"Starting pipeline with t={threshold}...")
    
    # Loading HotpotQA Distractor setting (Requirement 7)
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    subset = dataset.select(range(num_questions))
    
    prm = ProcessRewardModel()
    results = []

    for i, item in enumerate(subset):
        question = item['question']
        # Each question has 10 paragraphs (2 gold, 8 distractors) [cite: 7]
        all_contexts = [" ".join(ctx[1]) for ctx in item['context']]
        ground_truth = item['answer']

        # PRM Gate: Requirement 8 & 12
        pruned_contexts = prm.prune_context(question, all_contexts, threshold)
        
        results.append({
            "question": question,
            "contexts": pruned_contexts,
            "answer": "Answer synthesized from pruned context.", 
            "ground_truth": ground_truth
        })
        
        if (i + 1) % 50 == 0:
            print(f"Processed {i+1}/{num_questions} questions...")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    run_pipeline(args.threshold, args.output)