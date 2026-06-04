import torch
import random
import numpy as np
from sentence_transformers import CrossEncoder

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Global seed set to: {seed}")

class ProcessRewardModel:
    def __init__(self, model_name='cross-encoder/ms-marco-MiniLM-L-6-v2'):
        seed_everything(42)
        # Cross-encoders are excellent for multi-hop relevance scoring
        self.model = CrossEncoder(model_name)

    def score_step(self, question, context_paragraph):
        """Scores a single paragraph against the question."""
        # The model returns a raw logit; we convert it to a 0-1 probability
        score = self.model.predict([question, context_paragraph])
        return 1 / (1 + np.exp(-score))

    def prune_context(self, question, contexts, threshold):
        """
        Implementation of threshold pruning (Requirement 4).
        Only keeps paragraphs that score above 't'.
        """
        pruned_contexts = []
        for ctx in contexts:
            score = self.score_step(question, ctx)
            if score >= threshold:
                pruned_contexts.append(ctx)
        return pruned_contexts