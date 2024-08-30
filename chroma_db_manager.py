import chromadb
import uuid

class ChromaDBManager:
    def __init__(self, collection_name):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)

    def add_embeddings(self, embeddings, texts):
        ids = [str(uuid.uuid4()) for _ in range(len(texts))]  # Generate unique IDs for each text
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=[{"question": text} for text in texts]
        )

    def query_embeddings(self, query_embedding, n_results=5):
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
        return [result['question'] for result in results['metadatas'][0]]