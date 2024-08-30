import pandas as pd
from models.embedder import Embedder
from db.chroma_db import ChromaDBManager

def load_data(file_path):
    df = pd.read_csv(file_path)
    questions = pd.concat([df['question1'], df['question2']]).unique()
    return [q for q in questions if pd.notna(q)]

def main():
    # Load and preprocess the data
    file_path = 'datasets/questions.csv'
    questions = load_data(file_path)
    
    # Initialize the embedder and ChromaDB manager
    embedder = Embedder()
    chroma_db_manager = ChromaDBManager(collection_name="quora_questions")

    # Generate embeddings
    embeddings = embedder.encode(questions)

    # Store embeddings in ChromaDB
    chroma_db_manager.add_embeddings(embeddings, questions)

    # Example query
    query = "How can I lose weight fast?"
    query_embedding = embedder.encode_single(query)
    similar_questions = chroma_db_manager.query_embeddings(query_embedding)

    # Display the results
    for i, question in enumerate(similar_questions, 1):
        print(f"Similar Question {i}: {question}")

if __name__ == "__main__":
    main()
