from typing import List, Optional
from langchain_openai import OpenAIEmbeddings
from app.config import settings
from app.utils.langfuse_client import langfuse_client
from langfuse import observe

class TracedOpenAIEmbeddings(OpenAIEmbeddings):
    """
    A wrapper around OpenAIEmbeddings that automatically logs
    the embedding creation process to Langfuse.
    """
    
    @observe(as_type="span", name="embedding_generation")
    def embed_documents(self, texts: List[str], chunk_size: Optional[int] = None) -> List[List[float]]:
        langfuse_client.update_current_span(
            metadata={
                "num_chunks": len(texts),
                "embedding_model": self.model
            }
        )
        return super().embed_documents(texts, chunk_size)
        
    @observe(as_type="span", name="embedding_generation")
    def embed_query(self, text: str) -> List[float]:
        langfuse_client.update_current_span(
            metadata={
                "num_chunks": 1,
                "embedding_model": self.model
            }
        )
        return super().embed_query(text)

def get_embedder() -> TracedOpenAIEmbeddings:
    """
    Returns an instance of TracedOpenAIEmbeddings configured with
    settings from app.config.
    """
    return TracedOpenAIEmbeddings(
        model=settings.EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY
    )
