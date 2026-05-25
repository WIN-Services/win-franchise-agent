import os
import re
import json
import time
import pickle
import logging
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss
from openai import OpenAI, RateLimitError, APIConnectionError, APIStatusError
from tqdm import tqdm

# Setup logging
logger = logging.getLogger("embedding_pipeline")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

class EmbeddingPipeline:
    """
    A production-ready pipeline to process semantic chunks, format them semantically,
    generate vector embeddings using OpenAI, and persist them in FAISS and Pickle formats.
    """

    def __init__(self, api_key: str, model_name: str = "text-embedding-3-small"):
        if not api_key:
            raise ValueError("An OpenAI API Key must be provided.")
        
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name
        
    @staticmethod
    def load_json(file_path: str) -> List[Dict[str, Any]]:
        """Loads JSON raw data file containing list of semantic chunks."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Input file not found: {file_path}")
        
        logger.info(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            
        logger.info(f"Loaded {len(data)} chunk records.")
        return data

    @staticmethod
    def clean_text(text: str) -> str:
        """
        Cleans excessive whitespace, preserves meaningful formatting, and removes
        any duplicate repeated lines.
        """
        if not text:
            return ""
            
        lines = text.split("\n")
        cleaned_lines = []
        seen_lines = set()
        
        for line in lines:
            # Normalize line whitespace (collapse internal spaces)
            stripped_line = " ".join(line.split())
            
            if stripped_line:
                # Normalize to lowercase for exact deduplication matching
                norm_key = stripped_line.lower()
                if norm_key not in seen_lines:
                    seen_lines.add(norm_key)
                    cleaned_lines.append(stripped_line)
            else:
                # Preserve a single blank line for spacing but collapse multiple
                if cleaned_lines and cleaned_lines[-1] != "":
                    cleaned_lines.append("")
                    
        return "\n".join(cleaned_lines).strip()

    @staticmethod
    def humanize_key(key: str) -> str:
        """Converts a snake_case string (e.g., type_of_expenditure) to human-readable Capitalized string."""
        if not key:
            return ""
        return " ".join(word.capitalize() for word in key.split("_"))

    def generate_semantic_text(self, chunk: Dict[str, Any]) -> str:
        """
        Transforms a raw JSON chunk into a high-quality semantic text string.
        Supports both narrative text chunks and table row chunks.
        """
        meta = chunk.get("metadata", {})
        section = meta.get("section", "").strip()
        subsection = (meta.get("sub-section", "") or meta.get("subsection", "")).strip()
        
        # Table detection
        if "table" in chunk:
            table_slug = chunk["table"]
            # Table title should be Section Name or table slug if empty
            table_title = section if section else self.humanize_key(table_slug)
            parts = [f"Table: {table_title}"]
            
            # Process rest of attributes (the columns of the row)
            for key, value in chunk.items():
                if key in ("table", "metadata"):
                    continue
                    
                val_str = str(value).strip()
                if not val_str:
                    continue
                    
                human_k = self.humanize_key(key)
                cleaned_val = self.clean_text(val_str)
                
                # Long description elements should follow a newline separator
                if key.lower() in ("description", "content", "text", "details"):
                    parts.append(f"{human_k}:\n{cleaned_val}")
                else:
                    parts.append(f"{human_k}: {cleaned_val}")
            
            return "\n\n".join(parts)
        
        # Standard narrative text chunk
        else:
            text_val = chunk.get("text", "")
            cleaned_content = self.clean_text(text_val)
            
            parts = []
            if section:
                parts.append(f"Section: {section}")
            if subsection:
                parts.append(f"Subsection: {subsection}")
            if cleaned_content:
                parts.append(f"Content:\n{cleaned_content}")
                
            return "\n\n".join(parts)

    def embed_batch(self, batch_texts: List[str], max_retries: int = 5) -> List[List[float]]:
        """
        Generates embeddings for a batch of text strings using OpenAI API.
        Implements robust exponential backoff for error and rate-limit handling.
        """
        delay = 2
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model_name,
                    input=batch_texts
                )
                return [item.embedding for item in response.data]
            except (RateLimitError, APIConnectionError, APIStatusError) as e:
                if attempt == max_retries - 1:
                    logger.error(f"OpenAI Embedding failed after {max_retries} attempts. Final Error: {e}")
                    raise e
                logger.warning(f"OpenAI API error (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {delay}s...")
                time.sleep(delay)
                delay *= 2
            except Exception as e:
                logger.error(f"Unexpected error during embedding generation: {e}")
                raise e
        return []

    def generate_embeddings_pipeline(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """
        Iterates over the entire text corpus in batches to compute vector embeddings.
        Tracks progress using tqdm.
        """
        all_embeddings = []
        logger.info(f"Starting embedding generation for {len(texts)} units using '{self.model_name}' in batches of {batch_size}...")
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
            batch = texts[i : i + batch_size]
            try:
                batch_emb = self.embed_batch(batch)
                all_embeddings.extend(batch_emb)
            except Exception as e:
                logger.error(f"Batch processing terminated at index {i} due to a fatal exception: {e}")
                raise e
                
        # Convert to numpy float32 matrix for FAISS
        vectors = np.array(all_embeddings, dtype=np.float32)
        return vectors

    def build_faiss_index(self, vectors: np.ndarray) -> faiss.IndexFlatIP:
        """
        Constructs a FAISS IndexFlatIP.
        Normalizes vectors for strict Cosine Similarity compatibility.
        """
        if vectors.size == 0:
            raise ValueError("Cannot build index from empty vector array.")
            
        dimension = vectors.shape[1]
        logger.info(f"Initializing FAISS IndexFlatIP with dimension={dimension}.")
        
        index = faiss.IndexFlatIP(dimension)
        
        # L2 Normalize to turn inner product into Cosine Similarity
        logger.info("Normalizing vectors to unit L2 norm...")
        faiss.normalize_L2(vectors)
        
        index.add(vectors)
        logger.info(f"FAISS Index successfully built with {index.ntotal} vectors.")
        return index

    def save_artifacts(self, index: faiss.IndexFlatIP, pkl_data: List[Dict[str, Any]], output_dir: str):
        """Persists the generated FAISS vector file and python objects Pickle file to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        faiss_path = os.path.join(output_dir, "index.faiss")
        pkl_path = os.path.join(output_dir, "index.pkl")
        
        logger.info(f"Saving FAISS index to: {faiss_path}")
        faiss.write_index(index, faiss_path)
        
        logger.info(f"Saving Chunk Metadata Pickle to: {pkl_path}")
        with open(pkl_path, "wb") as f:
            pickle.dump(pkl_data, f)
            
        logger.info("Artifact storage complete!")
