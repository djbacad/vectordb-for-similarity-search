from sentence_transformers import SentenceTransformer
import torch

# Class to generate embeddings
class Embedder:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', device='cuda'):
        self.model = SentenceTransformer(model_name, device=device)

    # Method for single input
    def encode_single(self, text):
        return self.model.encode([text], convert_to_tensor=True)

    # Method for multiple inputs
    def encode(self, texts):
        return self.model.encode(texts, convert_to_tensor=True)
    
    