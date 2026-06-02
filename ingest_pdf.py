import os
import chromadb
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
DB_PATH = "my_vector_db"
COLLECTION_NAME = "lecture_knowledge"
# ---------------------

def ingest_file(file_path):
    print(f"📄 Reading file: {file_path}...")
    
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        print(f"   ... Found {len(text)} characters.")
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return

    # 2. Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    print(f"🔪 Split into {len(chunks)} chunks.")

    # 3. Embedding
    print("📥 Loading Embedding Model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 4. Database
    print("💾 Connecting to Database...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 5. Store
    print("🚀 Vectorizing and Storing...")
    ids = [f"{os.path.basename(file_path)}_chunk_{i}" for i in range(len(chunks))]
    embeddings = embedding_model.encode(chunks).tolist()
    
    collection.add(
        documents=chunks,
        embeddings=embeddings,
        ids=ids
    )
    print(f"✅ Success! Added {len(chunks)} new chunks.")

if __name__ == "__main__":
    target_file = input("Enter path to PDF file: ").strip()
    
    # --- FIX FOR MACOS TERMINAL INPUT ---
    # Remove quotes
    target_file = target_file.replace("'", "").replace('"', "")
    # Remove backslashes used to escape spaces
    target_file = target_file.replace(r"\ ", " ")
    # ------------------------------------
    
    if os.path.exists(target_file):
        ingest_file(target_file)
    else:
        print(f"❌ File not found: {target_file}")
