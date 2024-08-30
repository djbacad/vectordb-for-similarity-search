Similarity Search w/ Sentence Transformers and Chroma Vector DB
===============================================================

##### GIF Demos:
###### Storing embeddings to vector db
![dbstorergif](https://github.com/user-attachments/assets/716d78d5-5052-4505-9ae1-2d92bdb7b8dd)
###### Sample Run 1:
![run1](https://github.com/user-attachments/assets/173e71ba-44d7-4b09-9f74-27870893b494)
###### Sample Run 2:
![run2](https://github.com/user-attachments/assets/f91d09a9-48bd-4155-b41f-f9627407b9c9)

This project demonstrates how to perform similarity search using sentence embeddings with ChromaDB as the vector database. The example uses the Quora Question Pairs dataset to showcase how to embed text data and query for similar entries.

### Project Highlights:
- Sentence Embeddings: Utilizing the sentence-transformers library, specifically the all-MiniLM-L6-v2 model, to generate dense vector representations of text.
- Vector Database (ChromaDB): Storing and querying embeddings efficiently using ChromaDB.
- Persistent Storage: ChromaDB is configured to persistently store embeddings for future queries.

### Files Overview:
embedder.py: Handles the embedding of text data using a pre-trained SentenceTransformer model.
chroma_db_manager.py: Manages storing and querying embeddings in ChromaDB, with support for batch processing.
db_storer.py: Stores embeddings in ChromaDB persistently.
run.py: Queries ChromaDB for similar entries based on user input.
