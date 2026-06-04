import json, argparse, os, torch
from datasets import load_dataset
from prm import ProcessRewardModel, seed_everything
from retriever import MultiHopRetriever
from transformers import pipeline

def run_pipeline(threshold, output_file, num_questions=500):
    seed_everything(42) # Requirement 10: Fixed Seed
    print(f"🚀 Executing Advanced Pipeline | Threshold t={threshold}")
    
    dataset = load_dataset("hotpot_qa", "distractor", split="validation", trust_remote_code=True)
    subset = dataset.select(range(num_questions))
    
    prm = ProcessRewardModel()
    retriever = MultiHopRetriever()
    # LLM for reasoning steps (Requirement 2 & 9)
    llm = pipeline("text2text-generation", model="google/flan-t5-small", device=0 if torch.cuda.is_available() else -1)
    
    results = []
    for i, item in enumerate(subset):
        question, ground_truth = item['question'], item['answer']
        # Pool of 10 paragraphs (Requirement 7)
        all_paragraphs = [" ".join(ctx[1]) for ctx in item['context']]
        retriever.build_index(all_paragraphs)

        # --- HOP 1: Initial Search ---
        hop1_raw = retriever.retrieve(question, k=5)
        gold_hop1 = prm.prune_context(question, hop1_raw, threshold)
        
        # --- HOP 2: Multi-Hop Reasoning (Requirement 2 & 3) ---
        final_contexts = list(gold_hop1)
        if gold_hop1:
            context_so_far = " ".join(gold_hop1)
            # Ask the LLM what to look for next based on first clues
            bridge_prompt = f"Question: {question} Context found: {context_so_far} Next entity to search for:"
            sub_query = llm(bridge_prompt, max_new_tokens=20)[0]['generated_text']
            
            hop2_raw = retriever.retrieve(sub_query, k=3)
            gold_hop2 = prm.prune_context(sub_query, hop2_raw, threshold)
            # Combine unique contexts
            final_contexts = list(set(gold_hop1 + gold_hop2))

        # --- SYNTHESIS ---
        ctx_str = " ".join(final_contexts) if final_contexts else "No context found."
        ans_prompt = f"Answer using context. Context: {ctx_str} Question: {question}"
        
        results.append({
            "question": question, 
            "contexts": final_contexts,
            "answer": llm(ans_prompt, max_new_tokens=50)[0]['generated_text'],
            "ground_truth": ground_truth
        })
        if (i+1) % 50 == 0: print(f"✅ {i+1}/500 questions processed")

    with open(output_file, 'w') as f: json.dump(results, f, indent=4)