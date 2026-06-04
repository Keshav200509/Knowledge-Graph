import sys, types
# Patch to fix Ragas/Langchain internal import bug
m = types.ModuleType("langchain_community.chat_models.vertexai")
m.ChatVertexAI = type("ChatVertexAI", (), {})
sys.modules["langchain_community.chat_models.vertexai"] = m

import os, json, pandas as pd, numpy as np, argparse
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness
from datasets import Dataset

# [Insert your existing bootstrap_ci and run_evaluation functions here]