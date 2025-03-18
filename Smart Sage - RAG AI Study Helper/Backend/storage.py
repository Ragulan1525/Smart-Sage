import chromadb
from sentence_transformers import SentenceTransformer

# Initialize ChromaDB client
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("study_material")

# Load embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

def get_chroma_collection():
    """Initializes and returns a ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_db")  # Stores data persistently
    collection = client.get_or_create_collection(name="educational_data")
    return collection

def store_text_in_chroma(text, source, doc_id=None):
    """Stores extracted text in ChromaDB with embeddings."""
    if not text.strip():
        print("Warning: Empty text cannot be stored.")
        return

    embedding = model.encode(text).tolist()
    
    # Generate unique ID if not provided
    doc_id = doc_id or f"doc_{hash(text)}"

    collection.add(
        ids=[doc_id],  # Ensuring unique document ID
        documents=[text],
        embeddings=[embedding],
        metadatas=[{"source": source}]
    )
    print(f"Stored document in ChromaDB with ID: {doc_id}")
