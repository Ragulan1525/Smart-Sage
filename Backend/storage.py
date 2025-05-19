# import chromadb
# from sentence_transformers import SentenceTransformer
# import logging
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# # Initialize logging
# logging.basicConfig(level=logging.INFO)

# # Load embedding model globally
# model = SentenceTransformer('all-MiniLM-L6-v2')
# logging.info("✅ Embedding model loaded successfully!")

# def get_chroma_collection():
#     """Retrieves or creates the ChromaDB collection."""
#     try:
#         # Initialize ChromaDB client
#         client = chromadb.PersistentClient(path="./chroma_db")
#         # Retrieve or create a collection named "study_materials"
#         collection = client.get_or_create_collection(name="study_materials")
#         logging.info(f"✅ ChromaDB collection retrieved or created: {collection}")
#         return collection
#     except Exception as e:
#         logging.error(f"❌ Error getting ChromaDB collection: {e}")
#         return None

# def store_text_in_chroma_simple(text, filename):
#     """Stores extracted text into ChromaDB using chunking for better retrieval."""
#     try:
#         collection = get_chroma_collection()
#         if not collection:
#             raise ValueError("❌ ChromaDB collection is not available.")

#         text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
#         chunks = text_splitter.split_text(text)

#         if not chunks:
#             logging.warning(f"❌ No valid text chunks extracted from {filename}.")

#         logging.info(f"✅ Splitting resulted in {len(chunks)} chunks.")

#         embeddings = [model.encode(chunk).tolist() for chunk in chunks]
        
#         # Debug embeddings
#         if not embeddings or not all(embeddings):
#             logging.error("❌ Embeddings generation failed!")
#             return

#         collection.add(
#             documents=chunks,
#             embeddings=embeddings,
#             ids=[f"{filename}_{i}" for i in range(len(chunks))],
#             metadatas=[{"source": filename} for _ in chunks]
#         )

#         logging.info(f"✅ Stored {len(chunks)} chunks from {filename} into ChromaDB!")

#     except Exception as e:
#         logging.error(f"❌ Error storing text in ChromaDB: {e}")
#         raise

# # Example usage (for testing)
# if __name__ == "__main__":
#     test_text = "A doubly linked list is a data structure that consists of a sequence of elements, each containing a reference to the next and the previous element."
#     store_text_in_chroma_simple(test_text, "test_file.txt")











import chromadb
from sentence_transformers import SentenceTransformer
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Load embedding model globally
model = SentenceTransformer('all-MiniLM-L6-v2')
logging.info("✅ Embedding model loaded successfully!")

def get_chroma_collection():
    """Retrieves or creates the ChromaDB collection."""
    try:
        # Initialize ChromaDB client
        client = chromadb.PersistentClient(path="./chroma_db")
        # Retrieve or create a collection named "study_materials"
        collection = client.get_or_create_collection(name="study_materials")
        logging.info(f"✅ ChromaDB collection retrieved or created: {collection}")
        return collection
    except Exception as e:
        logging.error(f"❌ Error getting ChromaDB collection: {e}")
        return None

def store_text_in_chroma_simple(text, filename):
    """Stores extracted text into ChromaDB using chunking for better retrieval."""
    try:
        collection = get_chroma_collection()
        if not collection:
            raise ValueError("❌ ChromaDB collection is not available.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = text_splitter.split_text(text)

        if not chunks:
            logging.warning(f"❌ No valid text chunks extracted from {filename}.")

        logging.info(f"✅ Splitting resulted in {len(chunks)} chunks.")

        embeddings = [model.encode(chunk).tolist() for chunk in chunks]
        
        # Debug embeddings
        if not embeddings or not all(embeddings):
            logging.error("❌ Embeddings generation failed!")
            return

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=[f"{filename}_{i}" for i in range(len(chunks))],
            metadatas=[{"source": filename} for _ in chunks]
        )

        logging.info(f"✅ Stored {len(chunks)} chunks from {filename} into ChromaDB!")

    except Exception as e:
        logging.error(f"❌ Error storing text in ChromaDB: {e}")
        raise

# Example usage (for testing)
if __name__ == "__main__":
    test_text = "A doubly linked list is a data structure that consists of a sequence of elements, each containing a reference to the next and the previous element."
    store_text_in_chroma_simple(test_text, "test_file.txt")