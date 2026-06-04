import json
import argparse
import os
import torch
from datasets import load_dataset
from prm import ProcessRewardModel, seed_everything
from retriever import MultiHopRetriever
from transformers import pipeline

def run_pipeline(threshold, output_file, num_questions=500):
    seed_everything(42) # Requirement: Seed all randomness [cite: 16, 27]
    print(f"🚀 Initializing Multi-Hop Pipeline | Threshold t={threshold}")
    
    # Load HotpotQA Distractor set [cite: 11]
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    subset = dataset.select(range(num_questions))
    
    # Init components
    prm = ProcessRewardModel()
    retriever = MultiHopRetriever()
    
    # Synthesis & Sub-query LLM (Flan-T5 is efficient for CPU/Base GPU)
    llm = pipeline("text2text-generation", model="google/flan-t5-small", device=0 if torch.cuda.is_available() else -1)
    
    results = []

    for i, item in enumerate(subset):
        question = item['question']
        ground_truth = item['answer']
        # Combine title and text for each of the 10 paragraphs 
        all_paragraphs = [" ".join(ctx[1]) for ctx in item['context']]
        retriever.build_index(all_paragraphs)

        # --- HOP 1: Initial Search ---
        hop1_results = retriever.retrieve(question, k=5)
        pruned_hop1 = prm.prune_context(question, hop1_results, threshold) # Requirement 12
        
        # --- HOP 2: Bridge Reasoning ---
        final_contexts = list(pruned_hop1)
        if pruned_hop1:
            # Use LLM to decompose question into a bridge search term 
            context_so_far = " ".join(pruned_hop1)
            bridge_prompt = f"Question: {question} Context: {context_so_far} What specific person or entity should I look for next?"
            sub_query = llm(bridge_prompt, max_new_tokens=20)[0]['generated_text']
            
            hop2_results = retriever.retrieve(sub_query, k=3)
            pruned_hop2 = prm.prune_context(sub_query, hop2_results, threshold)
            
            # Combine unique contexts from both hops
            for ctx in pruned_hop2:
                if ctx not in final_contexts:
                    final_contexts.append(ctx)

        # --- FINAL SYNTHESIS ---
        context_text = " ".join(final_contexts) if final_contexts else "No relevant facts found."
        ans_prompt = f"Using the context provided, answer the question accurately. Context: {context_text} Question: {question}"
        generated_answer = llm(ans_prompt, max_new_tokens=50)[0]['generated_text']

        results.append({
            "question": question,
            "contexts": final_contexts,
            "answer": generated_answer,
            "ground_truth": ground_truth
        })

        if (i + 1) % 50 == 0:
            print(f"✅ Processed {i+1}/{num_questions} questions...")

    # Save results [cite: 18]
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"💾 Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    run_pipeline(args.threshold, args.output)