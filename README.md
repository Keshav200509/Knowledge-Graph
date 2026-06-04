# Knowledge-Graph

# Process Reward Model on Multi-Hop QA (AIMS DTU Research Intern 2026)

This repository contains a complete, publicly reproducible implementation of a **Process Reward Model (PRM)** designed to score intermediate reasoning steps in a multi-hop retrieval-augmented question-answering pipeline. The system is evaluated on the **HotpotQA distractor setting** across 500 held-out questions using the RAGAS framework with full statistical bootstrapping.

The core challenge from **Knowledge Graph.pdf** is navigating a pool of 10 paragraphs per question: 2 gold supporting facts connected via a hidden bridge entity, and 8 injected distractor paragraphs acting as semantic noise. The PRM acts as a strict logical gate at each retrieval hop to prune out distractors before final answer synthesis.

---

## 💻 Hardware Specifications & Runtime

*   **OS/Environment:** Linux (Ubuntu 22.04 LTS / GitHub Codespaces)
*   **CPU:** Intel(R) Xeon(R) @ 2.20GHz (4 Cores)
*   **GPU:** NVIDIA Tensor Core T4 (16GB VRAM)
*   **Total Execution Runtime:** ~38 minutes (includes pipeline runs and RAGAS LLM-as-a-judge metric evaluation for both ablations)

---

## 📂 Repository Structure

The project strictly follows the required layout outlined in **Knowledge Graph.pdf**:

```text
├── eval/
│   └── ragas_eval.py        # RAGAS evaluation script with 95% Bootstrap CIs
├── notebooks/
│   └── analysis.ipynb       # Failure analysis (hop-failures, false-positive prunes, charts)[cite: 1]
├── results/
│   ├── plots/               # Subfolder containing metric distribution charts[cite: 1]
│   ├── final_results_table.csv
│   ├── raw_04.json          # Pipeline traces for threshold t = 0.4[cite: 1]
│   └── raw_06.json          # Pipeline traces for threshold t = 0.6[cite: 1]
├── src/
│   ├── pipeline.py          # End-to-end orchestration (decompose, retrieve, gate, synthesize)[cite: 1]
│   ├── prm.py               # Process Reward Model definition, training loop, threshold pruning[cite: 1]
│   └── retriever.py         # FAISS index management & multi-hop retrieval controller[cite: 1]
├── README.md                # Environment specs, setup, and reproduction guide[cite: 1]
└── requirements.txt         # Fully pinned, locked version-compatible dependencies[cite: 1]
