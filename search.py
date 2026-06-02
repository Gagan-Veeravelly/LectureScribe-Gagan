import chromadb
from sentence_transformers import SentenceTransformer

# 1. Setup - Must match the model used to store
print("📥 Loading Embedding Model...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Connect to the existing database
# We point it to the folder created by the previous script
print("📂 Opening Vector Database...")
client = chromadb.PersistentClient(path="my_vector_db")

# Get the collection (must match the name used in store_vectors.py)
# If you didn't change it, it's likely named "scraped_data" or "knowledge_base"
# Let's try "knowledge_base" first, or list all to be safe.
try:
    collection = client.get_collection(name="knowledge_base")
except:
    # If the name is different, let's grab the first available collection
    cols = client.list_collections()
    if cols:
        collection = cols[0]
        print(f"⚠️ Found collection named: '{collection.name}'")
    else:
        print("❌ No collections found in database.")
        exit()

while True:
    print("\n" + "="*40)
    query_text = input("🔍 Ask a question (or type 'exit'): ")
    
    if query_text.lower() == 'exit':
        break

    # 3. Convert query to vector
    query_vector = embedding_model.encode(query_text).tolist()

    # 4. Search the database
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=2  # Return top 2 matches
    )

    # 5. Display results
    print("\n✅ Found relevant info:")
    for i, doc in enumerate(results['documents'][0]):
        print(f"\n--- Result {i+1} ---")
        print(doc)
