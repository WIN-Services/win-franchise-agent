import tiktoken
from typing import List, Dict, Any

class TokenChunker:
    """Chunks text into token-based limits with overlap."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, encoding_name: str = "cl100k_base"):
        """
        Initializes the chunker.
        chunk_size: Target token size for each chunk (req: 300-800)
        chunk_overlap: Overlap between chunks in tokens (req: 50-100)
        encoding_name: The tiktoken encoding to use (default: cl100k_base for OpenAI)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.get_encoding(encoding_name)
        
    def chunk(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Takes a list of documents (with 'text' and 'metadata') and returns chunked documents.
        """
        chunked_docs = []
        
        for doc in documents:
            text = doc.get("text", "")
            base_metadata = doc.get("metadata", {})
            
            tokens = self.encoding.encode(text)
            
            i = 0
            chunk_index = 0
            while i < len(tokens):
                end = min(i + self.chunk_size, len(tokens))
                chunk_tokens = tokens[i:end]
                chunk_text = self.encoding.decode(chunk_tokens)
                
                # Prepend subheading/section to the chunk text so the embedding and agent sees it
                subheading = base_metadata.get('section', 'General Content')
                source = base_metadata.get('source', 'Unknown Source')
                
                context_prefix = f"Source: {source}\nSection: {subheading}\n\n"
                
                # Create a copy of metadata and add chunk specifics
                metadata = base_metadata.copy()
                metadata["chunk_index"] = chunk_index
                
                chunked_docs.append({
                    "text": context_prefix + chunk_text,
                    "metadata": metadata
                })
                
                if end == len(tokens):
                    break
                    
                i += self.chunk_size - self.chunk_overlap
                chunk_index += 1
                
        return chunked_docs
