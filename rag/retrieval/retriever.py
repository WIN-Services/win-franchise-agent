from typing import List, Dict, Any
from langfuse import observe
from app.utils.langfuse_client import langfuse_client
from rag.embeddings.store import VectorStoreManager
from app.config import settings

class FranchiseRetriever:
    """
    Handles retrieval of relevant chunks from the FAISS vector store.
    """
    
    def __init__(self):
        self.manager = VectorStoreManager()
        
    @observe(name="retrieval")
    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        Converts query to embedding and retrieves Top-K chunks.
        Logs query, top chunks, and metadata to Langfuse.
        """
        top_k = k or settings.TOP_K_RESULTS
        
        try:
            retriever = self.manager.get_retriever(k=top_k)
            docs = retriever.invoke(query)
        except ValueError:
            # Vector store not initialized
            docs = []
            
        # Format the output chunks
        results = []
        for doc in docs:
            results.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
            
        # Log to Langfuse span
        langfuse_client.update_current_span(
            input={"query": query, "top_k": top_k},
            output={"retrieved_chunks": results}
        )
        
        return results
