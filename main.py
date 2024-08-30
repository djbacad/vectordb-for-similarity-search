from embedder import Embedder
from chroma_db_manager import ChromaDBManager
import pandas as pd
import warnings
import torch

# Suppress all warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_data(file_path):
    df = pd.read_csv(file_path)
    questions = pd.concat([df['question1'], df['question2']]).unique()
    return [q for q in questions if pd.notna(q)]

def main():
    # Load and preprocess the data
    file_path = 'datasets/questions_sampled.csv'
    questions = load_data(file_path)
    
    # Initialize the embedder and ChromaDB manager
    print("Initializing Embedder...⌛")
    embedder = Embedder(model_name='sentence-transformers/all-MiniLM-L6-v2')
    print("Starting ChromaDB Manager...⌛")
    chroma_db_manager = ChromaDBManager(collection_name="quora_questions")

    # Generate embeddings
    print("Generating Embeddings in ChromaDB...⌛")
    embeddings = embedder.encode(questions)

    # Convert embeddings from a list of tensors to a list of lists (required by chromadb)
    if isinstance(embeddings, list) and isinstance(embeddings[0], torch.Tensor):
        embeddings = [e.tolist() for e in embeddings]
  
    # Store embeddings in ChromaDB
    print("Storing Embeddings in ChromaDB...⌛")
    chroma_db_manager.add_embeddings(embeddings, questions)

    # Example query
    print("Sample Query...⌛")
    query = "How can I lose weight fast?"
    query_embedding = embedder.encode_single(query)
    similar_questions = chroma_db_manager.query_embeddings(query_embedding)

    # Display the results
    for i, question in enumerate(similar_questions, 1):
        print(f"Similar Question {i}: {question}")

    print ("End")
if __name__ == "__main__":
    main()
