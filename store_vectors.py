import chromadb
from sentence_transformers import SentenceTransformer
import os

# 1. Setup
# We use a free, powerful model to turn text into numbers
# "all-MiniLM-L6-v2" is fast and runs perfectly on a Mac
print("📥 Loading Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Setup ChromaDB (This creates a folder 'my_vector_db' to store data)
chroma_client = chromadb.PersistentClient(path="my_vector_db")
collection = chroma_client.get_or_create_collection(name="lecture_knowledge")

def chunk_text(text, chunk_size=500):
    """Splits huge text into smaller digestible pieces for the AI."""
    chunks = []
    # Simple splitting (in production, use LangChain splitters)
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

def store_lecture_data():
    # 2. Load your specific file
    # I am using the filename from your logs. 
    filename = "Operating Systems - Lecture 5 - Process Synchronization_summary.md"
    
    if not os.path.exists(filename):
        print(f"❌ Error: Could not find {filename}")
        print("Please check if the file is in the same folder as this script.")
        return

    with open(filename, "r", encoding="utf-8") as f:
        full_text = f.read()

    print("🔪 Chunking text...")
    chunks = chunk_text(full_text)
    
    print(f"📊 Converting {len(chunks)} chunks into vectors (math)...")
    embeddings = embedding_model.encode(chunks)
    
    print("💾 Saving to Vector Database...")
    # Add to ChromaDB
    collection.add(
        documents=chunks,
        embeddings=embeddings.tolist(),
        ids=[f"id_{i}" for i in range(len(chunks))]
    )
    
    print(f"✅ Success! Stored {len(chunks)} knowledge chunks in 'my_vector_db'.")

if __name__ == "__main__":
    store_lecture_data()
