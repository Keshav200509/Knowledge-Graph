import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class MultiHopRetriever:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        # Using a standard bi-encoder for efficient FAISS indexing
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.paragraphs = []

    def build_index(self, paragraphs):
        """Creates a FAISS index for the 10 paragraphs in the distractor set."""
        self.paragraphs = paragraphs
        embeddings = self.model.encode(paragraphs)
        dimension = embeddings.shape[1]
        
        # L2 distance index for similarity search
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def retrieve(self, query, k=3):
        """Finds top-k paragraphs for the current reasoning step."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")
            
        query_vec = self.model.encode([query])
        distances, indices = self.index.search(np.array(query_vec).astype('float32'), k)
        
        return [self.paragraphs[i] for i in indices[0]]