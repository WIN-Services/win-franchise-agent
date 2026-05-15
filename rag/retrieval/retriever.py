from typing import List, Dict, Any
import logging
from langfuse import observe
from app.utils.langfuse_client import langfuse_client
from rag.embeddings.store import VectorStoreManager
from rag.retrieval.reranker import FranchiseReranker
from app.config import settings

logger = logging.getLogger("retriever")

class FranchiseRetriever:
    """
    Handles retrieval of relevant chunks from the FAISS vector store
    and enforces highly-accurate re-scoring using the CrossEncoder.
    """
    
    def __init__(self):
        self.manager = VectorStoreManager()
        try:
            self.reranker = FranchiseReranker()
        except Exception as e:
            logger.error(f"Failed to load cross-encoder reranker: {e}. Initial retrieval might skip rerank stage.")
            self.reranker = None
        
    @observe(name="retrieval")
    def retrieve(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """
        1. Converts query to embedding and retrieves candidate chunks (top-20 by default).
        2. Reranks the candidates down to the final set (top-5 by default).
        Logs query, results, and metadata to Langfuse.
        """
        # The base FAISS retrieval depth
        initial_k = settings.TOP_K_RESULTS
        # The target output depth after re-scoring
        target_k = k or settings.TOP_K_RERANK
        
        try:
            retriever = self.manager.get_retriever(k=initial_k)
            docs = retriever.invoke(query)
        except ValueError:
            # Vector store not initialized
            docs = []
            
        # Format the raw output candidate chunks
        candidates = []
        for doc in docs:
            candidates.append({
                "text": doc.page_content,
                "metadata": doc.metadata
            })
            
        logger.info(f"Retrieved {len(candidates)} raw candidates from FAISS vector store.")
        
        # Apply cross-encoder reranking if initialized and chunks exist
        if self.reranker and candidates:
            results = self.reranker.rerank(query=query, chunks=candidates, top_n=target_k)
        else:
            # Fallback to raw slices
            results = candidates[:target_k]
            
        # Log the enriched and filtered telemetry to Langfuse span
        langfuse_client.update_current_span(
            input={"query": query, "initial_k": initial_k, "target_k": target_k},
            output={"retrieved_chunks": results}
        )
        
        return results
