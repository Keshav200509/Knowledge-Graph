import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MultiHopRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.paragraphs = []

    def build_index(self, paragraphs):
        """Requirement 1: Build FAISS index for the 10 paragraph pool."""
        self.paragraphs = paragraphs
        embeddings = self.model.encode(paragraphs)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query, k=3):
        """Requirement 1: Return top-K relevant paragraphs using vector search."""
        if self.index is None: return []
        query_vec = self.model.encode([query])
        # Search the index for the closest k paragraphs
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        return [self.paragraphs[i] for i in indices[0]]