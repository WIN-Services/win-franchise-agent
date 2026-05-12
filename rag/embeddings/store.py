import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from app.config import settings
from rag.embeddings.generator import get_embedder

class VectorStoreManager:
    """
    Manages the FAISS vector store, handling initialization,
    adding documents, and saving/loading the index to/from disk.
    """
    
    def __init__(self):
        self.embedder = get_embedder()
        self.vector_store_path = settings.VECTOR_DB_PATH
        self.vector_store = None
        
    def load_or_create_index(self):
        """
        Loads the FAISS index from disk if it exists, otherwise initializes an empty one.
        """
        if os.path.exists(self.vector_store_path) and os.path.isdir(self.vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    folder_path=self.vector_store_path, 
                    embeddings=self.embedder,
                    allow_dangerous_deserialization=True # required for loading local FAISS in newer langchain versions
                )
                print(f"Loaded existing FAISS index from {self.vector_store_path}")
            except Exception as e:
                print(f"Error loading FAISS index: {e}. Will create a new one upon adding documents.")
                self.vector_store = None
        else:
            print("No existing FAISS index found. A new one will be created when documents are added.")

    def add_documents(self, documents: List[Document]):
        """
        Adds a list of Langchain Documents to the FAISS index and saves it to disk.
        """
        if not documents:
            return
            
        if self.vector_store is None:
            # Attempt to load first if not explicitly loaded
            self.load_or_create_index()
            
        if self.vector_store is None:
            # Create a new index from documents
            self.vector_store = FAISS.from_documents(documents, self.embedder)
        else:
            # Add to existing index
            self.vector_store.add_documents(documents)
            
        self.save_index()
        
    def save_index(self):
        """
        Saves the current FAISS index to disk.
        """
        if self.vector_store is not None:
            # Create directory if it doesn't exist
            os.makedirs(self.vector_store_path, exist_ok=True)
            self.vector_store.save_local(self.vector_store_path)
            print(f"Saved FAISS index to {self.vector_store_path}")

    def get_retriever(self, k: int = None):
        """
        Returns a retriever interface for the vector store.
        """
        if self.vector_store is None:
            self.load_or_create_index()
            
        if self.vector_store is None:
            raise ValueError("Vector store is not initialized. Add documents first.")
            
        search_kwargs = {"k": k or settings.TOP_K_RESULTS}
        return self.vector_store.as_retriever(search_kwargs=search_kwargs)
