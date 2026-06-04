import json, argparse, os, torch
from datasets import load_dataset
from prm import ProcessRewardModel, seed_everything
from retriever import MultiHopRetriever
from transformers import pipeline

def run_pipeline(threshold, output_file, num_questions=500):
    seed_everything(42) # Requirement: Report and seed all randomness
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    subset = dataset.select(range(num_questions))
    
    prm, retriever = ProcessRewardModel(), MultiHopRetriever()
    # LLM for Bridge-entity extraction and Synthesis
    llm = pipeline("text2text-generation", model="google/flan-t5-small", device=0 if torch.cuda.is_available() else -1)
    
    results = []
    for i, item in enumerate(subset):
        question, ground_truth = item['question'], item['answer']
        retriever.build_index([" ".join(ctx[1]) for ctx in item['context']])

        # HOP 1
        hop1_raw = retriever.retrieve(question, k=5)
        gold_hop1 = prm.prune_context(question, hop1_raw, threshold)
        
        # HOP 2 (The Multi-Hop Bridge Entity)
        final_contexts = list(gold_hop1)
        if gold_hop1:
            bridge_prompt = f"Question: {question} Context: {' '.join(gold_hop1)} Next entity:"
            sub_query = llm(bridge_prompt, max_new_tokens=20)[0]['generated_text']
            gold_hop2 = prm.prune_context(sub_query, retriever.retrieve(sub_query, k=3), threshold)
            final_contexts = list(set(gold_hop1 + gold_hop2))

        # SYNTHESIS (Essential for RAGAS correctness)
        ctx_text = " ".join(final_contexts) if final_contexts else "No context found."
        results.append({
            "question": question, "contexts": final_contexts,
            "answer": llm(f"Context: {ctx_text} Question: {question}", max_new_tokens=50)[0]['generated_text'],
            "ground_truth": ground_truth
        })
        if (i+1) % 50 == 0: print(f"✅ {i+1}/500 Processed")

    with open(output_file, 'w') as f: json.dump(results, f, indent=4)