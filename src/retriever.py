import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MultiHopRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.paragraphs = []

    def build_index(self, paragraphs):
        """Encodes and indexes the 10 paragraphs from the distractor set."""
        self.paragraphs = paragraphs
        embeddings = self.model.encode(paragraphs)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query, k=3):
        """Returns the top K most similar paragraphs for the query."""
        if self.index is None: return []
        query_vec = self.model.encode([query])
        _, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        return [self.paragraphs[i] for i in indices[0]]