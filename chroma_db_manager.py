import chromadb

class ChromaDBManager:
    def __init__(self, collection_name):
        self.client = chromadb.Client()
        self.collection = self.client.create_collection(name=collection_name)

    def add_embeddings(self, embeddings, texts):
        self.collection.add(
            embeddings=embeddings.tolist(),
            metadatas=[{"question": text} for text in texts]
        )

    def query_embeddings(self, query_embedding, n_results=5):
        results = self.collection.query(query_embeddings=query_embedding.tolist(), n_results=n_results)
        return [result['question'] for result in results['metadatas'][0]]