import re
import unicodedata
import tiktoken
import numpy as np
from typing import List, Dict, Any, Optional
from app.config import settings
from rag.embeddings.generator import get_embedder

class SemanticChunker:
    """
    Intelligently splits text based on semantic shifts, detected via OpenAI embeddings.
    Also maintains token overlap across semantic boundaries as specified in config.
    """
    
    def __init__(
        self, 
        overlap_tokens: int = None, 
        target_max_tokens: int = None,
        buffer_size: int = 1,
        breakpoint_threshold_percentile: int = 95,
        encoding_name: str = "cl100k_base"
    ):
        """
        Args:
            overlap_tokens: Number of tokens from the previous chunk to overlap. Defaults to settings.CHUNK_OVERLAP.
            target_max_tokens: Max tokens for a chunk before we consider splitting. Defaults to settings.CHUNK_SIZE.
            buffer_size: Number of adjacent sentences to combine for contextual embedding comparisons.
            breakpoint_threshold_percentile: The percentile threshold of semantic distances to consider a topic shift.
            encoding_name: tiktoken encoding model (defaults to cl100k_base for OpenAI).
        """
        self.overlap_tokens = overlap_tokens if overlap_tokens is not None else settings.CHUNK_OVERLAP
        self.target_max_tokens = target_max_tokens if target_max_tokens is not None else settings.CHUNK_SIZE
        self.buffer_size = buffer_size
        self.breakpoint_threshold_percentile = breakpoint_threshold_percentile
        
        self.encoding = tiktoken.get_encoding(encoding_name)
        # Get traced embedder from your RAG generator
        self.embedder = get_embedder()

    def _clean_text(self, text: str) -> str:
        """
        Normalizes Unicode characters. Replaces curly quotes, smart apostrophes,
        and special dashes with standard ASCII equivalents to ensure clean processing.
        """
        if not text:
            return ""
            
        # Standard Unicode NFKC normalization (handles accents, ligatures, etc)
        text = unicodedata.normalize("NFKC", text)
        
        # Map specific common non-ASCII punctuation to their clean standard equivalents
        replacements = {
            '\u201c': '"',  # Left Double Quotation Mark “
            '\u201d': '"',  # Right Double Quotation Mark ”
            '\u2018': "'",  # Left Single Quotation Mark ‘
            '\u2019': "'",  # Right Single Quotation Mark ’ (apostrophe)
            '\u201a': "'",  # Single Low-9 Quotation Mark
            '\u201b': "'",  # Single High-Reversed-9 Quotation Mark
            '\u201e': '"',  # Double Low-9 Quotation Mark
            '\u201f': '"',  # Double High-Reversed-9 Quotation Mark
            '\u2013': '-',  # En dash –
            '\u2014': '-',  # Em dash —
            '\u2026': '...', # Horizontal Ellipsis …
            '\u00a0': ' ',  # Non-breaking space
        }
        
        for orig, repl in replacements.items():
            text = text.replace(orig, repl)
            
        return text

    def _split_to_sentences(self, text: str) -> List[str]:
        """
        Splits text into logical sentences using a regex, preserving reasonable structures.
        """
        # Normalize whitespaces slightly but keep paragraph markers if short
        text = text.strip()
        if not text:
            return []
        
        # Standard sentence splitter regex: looks for ending punctuation followed by space/newlines
        # Also handles markdown lists as independent units
        raw_sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
        
        # Filter and clean
        sentences = [s.strip() for s in raw_sentences if s.strip()]
        return sentences

    def _get_cosine_distances(self, embeddings: np.ndarray) -> List[float]:
        """
        Computes cosine distance between consecutive embedding rows.
        """
        distances = []
        for i in range(len(embeddings) - 1):
            emb1 = embeddings[i]
            emb2 = embeddings[i + 1]
            
            # L2 normalized vectors allow easy cosine similarity by dot product
            norm1 = np.linalg.norm(emb1)
            norm2 = np.linalg.norm(emb2)
            
            if norm1 == 0 or norm2 == 0:
                similarity = 0.0
            else:
                similarity = np.dot(emb1, emb2) / (norm1 * norm2)
            
            # Cosine distance = 1 - similarity
            distance = 1.0 - similarity
            distances.append(distance)
            
        return distances

    def chunk_text(self, text: str) -> List[str]:
        """
        Main semantic chunking logic for a single string block.
        Returns list of chunked text.
        """
        # Sanitize unicodes, curly quotes, and smart apostrophes
        text = self._clean_text(text)
        
        # Check if the whole text already fits within target token count
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.target_max_tokens:
            return [text] if text.strip() else []

        sentences = self._split_to_sentences(text)
        
        # If very short in sentences, keep as a single chunk
        if len(sentences) <= 2:
            return [text] if text.strip() else []

        # 1. Group sentences into buffer windows to capture smoother semantic signal
        combined_sentences = []
        for i in range(len(sentences)):
            start = max(0, i - self.buffer_size)
            end = min(len(sentences), i + self.buffer_size + 1)
            
            # Combine adjacent sentences for context
            window = sentences[start:end]
            combined_sentences.append(" ".join(window))
            
        # 2. Compute embeddings for these combined sentence groups
        try:
            embeddings = self.embedder.embed_documents(combined_sentences)
            embeddings = np.array(embeddings)
        except Exception as e:
            # Fallback in case API fails: just return original text (or we could raise)
            print(f"Error generating embeddings in SemanticChunker: {e}")
            return [text]

        # 3. Compute consecutive cosine distances
        distances = self._get_cosine_distances(embeddings)
        
        if not distances:
            return [text]

        # 4. Find breakpoints based on distance percentile
        threshold = np.percentile(distances, self.breakpoint_threshold_percentile)
        
        # Detect boundary indices (after which sentence index we cut)
        breakpoint_indices = []
        for idx, dist in enumerate(distances):
            if dist > threshold:
                breakpoint_indices.append(idx)
                
        # 5. Assemble the sentences into semantic groups
        groups = []
        start_idx = 0
        for breakpoint in breakpoint_indices:
            groups.append(sentences[start_idx:breakpoint + 1])
            start_idx = breakpoint + 1
        # Add last group
        groups.append(sentences[start_idx:])
        
        # 6. Form text chunks and inject token overlap
        text_chunks = []
        
        for i, group in enumerate(groups):
            chunk_body = " ".join(group).strip()
            if not chunk_body:
                continue
                
            # If this is NOT the first chunk, and overlap is requested, prepend overlap
            if i > 0 and self.overlap_tokens > 0:
                # Get raw content from the previous chunk to extract overlap
                prev_chunk = " ".join(groups[i - 1]).strip()
                prev_tokens = self.encoding.encode(prev_chunk)
                
                # Get last 'overlap_tokens' tokens
                overlap_segment_tokens = prev_tokens[-self.overlap_tokens:]
                overlap_text = self.encoding.decode(overlap_segment_tokens)
                
                # Prepend overlap text with indicator or just seamless concatenation
                # Using a seamless concatenation, usually marked or just direct.
                final_chunk_text = f"{overlap_text} {chunk_body}".strip()
            else:
                final_chunk_text = chunk_body
                
            text_chunks.append(final_chunk_text)
            
        return text_chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Similar interface to TokenChunker. Takes list of dicts with 'text' and 'metadata'.
        Yields chunks with metadata injected and indexed.
        """
        results = []
        for doc in documents:
            content = doc.get("text", "")
            base_metadata = doc.get("metadata", {})
            
            chunks = self.chunk_text(content)
            
            for idx, chunk in enumerate(chunks):
                meta = base_metadata.copy()
                meta["chunk_index"] = idx
                meta["chunking_strategy"] = "semantic"
                
                results.append({
                    "text": chunk,
                    "metadata": meta
                })
                
        return results
