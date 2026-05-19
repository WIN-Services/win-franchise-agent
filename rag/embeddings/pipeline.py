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

    @staticmethod
    def determine_topic(section: str, subsection: str, text: str) -> str:
        """
        Determines the thematic topic based on metadata and content keywords.
        """
        sec_sub = (section + " " + subsection).lower()
        text_lower = text.lower()
        
        if "veteran" in sec_sub or "first responder" in sec_sub:
            return "Veterans & First Responders Program"
        if "item 19" in sec_sub or "gross revenue" in sec_sub or "earnings" in sec_sub:
            return "Item 19 Financial Performance"
        if "fee" in sec_sub or "cost" in sec_sub or "investment" in sec_sub or "expenditure" in sec_sub or "pricing" in sec_sub:
            return "Initial Investment & Fees"
        if "train" in sec_sub or "bootcamp" in sec_sub or "academy" in sec_sub:
            return "Training & Support Programs"
        if "market" in sec_sub or "advertis" in sec_sub or "brand" in sec_sub:
            return "Marketing & Advertising Support"
        if "technolog" in sec_sub or "software" in sec_sub or "app" in sec_sub or "concierge" in sec_sub:
            return "Technology & Systems"
        if "territory" in sec_sub or "location" in sec_sub or "area" in sec_sub:
            return "Territory & Location Selection"
        if "faq" in sec_sub or "frequently asked" in sec_sub:
            return "General FAQ"
        if "disclosure" in sec_sub or "fdd" in sec_sub or "agreement" in sec_sub or "contract" in sec_sub or "legal" in sec_sub:
            return "Legal Disclosures & FDD"
        if "step" in sec_sub or "onboarding" in sec_sub or "launch" in sec_sub:
            return "Franchise Agreement & Steps"
            
        # Try matching text keywords
        if "veteran" in text_lower or "first responder" in text_lower:
            return "Veterans & First Responders Program"
        if "item 19" in text_lower or "gross revenue" in text_lower:
            return "Item 19 Financial Performance"
        if "fee" in text_lower or "cost" in text_lower or "investment" in text_lower:
            return "Initial Investment & Fees"
        if "training" in text_lower or "bootcamp" in text_lower:
            return "Training & Support Programs"
        if "marketing" in text_lower or "advertising" in text_lower:
            return "Marketing & Advertising Support"
        
        # Fallback to subsection or section if available
        if subsection:
            return subsection
        if section:
            return section
        return "Overview"

    def generate_semantic_text(self, chunk: Dict[str, Any]) -> str:
        """
        Transforms a raw JSON chunk into a high-quality semantic text string.
        Adheres strictly to the requested semantic structure.
        """
        meta = chunk.get("metadata", {})
        section = meta.get("section", "").strip()
        subsection = (meta.get("sub-section", "") or meta.get("subsection", "")).strip()
        raw_ct = meta.get("content_type", "")
        if isinstance(raw_ct, list):
            content_type = ", ".join(str(c).strip() for c in raw_ct if str(c).strip())
        else:
            content_type = str(raw_ct).strip()
        source_val = meta.get("source", "").strip()
        table_name = chunk.get("table", "").strip() if "table" in chunk else ""
        
        # Build main semantic content
        if "table" in chunk:
            row_parts = []
            for key, value in chunk.items():
                if key in ("table", "metadata", "display_text"):
                    continue
                val_str = str(value).strip()
                if not val_str:
                    continue
                human_k = self.humanize_key(key)
                cleaned_val = self.clean_text(val_str)
                if key.lower() in ("description", "content", "text", "details"):
                    row_parts.append(f"{human_k}:\n{cleaned_val}")
                else:
                    row_parts.append(f"{human_k}: {cleaned_val}")
            main_content = "\n\n".join(row_parts)
        else:
            text_val = chunk.get("text", "")
            main_content = self.clean_text(text_val)
            
        topic = self.determine_topic(section, subsection, main_content)
        
        header_lines = [
            f"Section: {section}",
            f"Subsection: {subsection}",
            f"Content Type: {content_type}",
            f"source: {source_val}",
            f"Table Name: {table_name}",
            f"Topic: {topic}"
        ]
        
        headers_str = "\n".join(header_lines)
        return f"{headers_str}\n\n{main_content}"

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
