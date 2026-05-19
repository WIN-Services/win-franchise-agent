import torch
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder
from app.config import settings

logger = logging.getLogger("reranker")

# Suppress HuggingFace Hub nagging warnings regarding unauthenticated Hub requests
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

class FranchiseReranker:
    """
    Reranks retrieved document chunks using a cross-encoder architecture
    to yield more accurate semantic ordering.
    """

    def __init__(self):
        self.model_name = settings.RERANK_MODEL
        
        # Device detection for acceleration
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
            
        logger.info(f"Initializing CrossEncoder Reranker ('{self.model_name}') on device: {self.device}")
        
        try:
            # Initialize CrossEncoder
            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
                max_length=512
            )
            logger.info("Reranker model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load rerank model: {e}")
            raise e

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = None) -> List[Dict[str, Any]]:
        """
        Computes semantic similarity scores between query and candidates,
        then re-orders and trims down list to the most relevant ones.
        
        Args:
            query: The user prompt
            chunks: Retreived chunks containing "text" and "metadata" keys.
            top_n: Target size of returning payload. Defaults to settings.TOP_K_RERANK.
        """
        if not chunks:
            return []
            
        top_n = top_n or settings.TOP_K_RERANK
        
        # Construct query-document pairs for Cross-Encoder scoring
        pairs = [[query, chunk.get("text", "")] for chunk in chunks]
        
        logger.info(f"Computing cross-encoder scores for {len(pairs)} candidate pairs...")
        
        try:
            # Predict yields numerical scores
            scores = self.model.predict(pairs, batch_size=16)
            
            # Map scores to the input structures
            for i, score in enumerate(scores):
                chunks[i]["rerank_score"] = float(score)
                
            # Sort descendently by rerank score
            ranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            
            logger.info(f"Reranking completed. Returning top {min(len(ranked_chunks), top_n)} items.")
            return ranked_chunks[:top_n]
            
        except Exception as e:
            logger.error(f"Error encountered during reranking execution: {e}. Falling back to unranked results.")
            # Return fallback slice if scoring fails
            return chunks[:top_n]
