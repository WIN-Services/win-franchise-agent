import os
import sys
import re
import json
import logging
from typing import List, Dict, Any, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Append project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from rag.embeddings.pipeline import EmbeddingPipeline

# Setup script logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def normalize_text(text: str) -> str:
    """
    Normalizes text strictly for exact deduplication detection:
    - Lowercase
    - Collapses repeated punctuation characters (e.g. '...' -> '.', '!!!' -> '!')
    - Strips non-alphanumeric or control gaps
    - Collapses all excessive whitespace and newlines into a single space.
    """
    if not text:
        return ""
        
    # 1. Convert to lower
    norm = text.lower()
    
    # 2. Repeated punctuation collapse
    # Matches any non-word character \W followed by one or more of itself
    norm = re.sub(r'(\W)\1+', r'\1', norm)
    
    # 3. Whitespace collapse and newline removal
    norm = " ".join(norm.split())
    
    return norm

def extract_numbers(text: str) -> Set[str]:
    """
    Extracts all numerical representations from a string (e.g. '$18,000', '35+', '2026').
    This helps in creating a rigorous safeguard to prevent merging different financial values or dates.
    """
    if not text:
        return set()
    # Extracts digits potentially mixed with commas, dollar signs, decimals
    pattern = r'\$?\b\d+(?:,\d{3})*(?:\.\d+)?%?\b'
    return set(re.findall(pattern, text))

def should_preserve_variant(c1: Dict[str, Any], c2: Dict[str, Any]) -> bool:
    """
    Safeguard heuristic function. Returns True if we MUST preserve both chunks.
    Conditions for preserving:
    1. One is a table and the other isn't.
    2. They contain different specific numeric values/dollar amounts/dates.
    3. The section contents significantly differ.
    """
    # Safeguard 1: Cross-schema checks (Table vs Narrative text)
    if ("table" in c1) != ("table" in c2):
        return True
        
    # Safeguard 2: Numbers/Amounts comparison (Extremely important for Item 19, fees, years)
    nums1 = extract_numbers(c1["semantic_text"])
    nums2 = extract_numbers(c2["semantic_text"])
    if nums1 != nums2:
        # The sets of numbers found differ - do not deduplicate.
        return True
        
    # Safeguard 3: Table rows (if both are tables, verify they are actually representing the same table & metadata)
    if "table" in c1 and "table" in c2:
        if c1["table"] != c2["table"]:
            return True
            
    return False

def run_deduplication(
    input_path: str, 
    output_path: str, 
    near_dup_threshold: float = 0.88
):
    # Setup temporary pipeline to utilize semantic text constructor
    # Dummy API key is fine as we are not hitting OpenAI during deduplication
    pipeline = EmbeddingPipeline(api_key="dummy_key")
    
    logger.info(f"Starting deduplication pipeline loading from '{input_path}'...")
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            raw_chunks = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return
        
    total_loaded = len(raw_chunks)
    logger.info(f"Step 1: Loaded {total_loaded} chunks. Constructing semantic representation...")
    
    # Generate semantic_text for every record for precise analysis
    processed_records = []
    for i, chunk in enumerate(raw_chunks):
        sem_text = pipeline.generate_semantic_text(chunk)
        processed_records.append({
            "index": i,
            "semantic_text": sem_text,
            "normalized_text": normalize_text(sem_text),
            "original_data": chunk
        })
        
    # ====================================================
    # PHASE 1: EXACT DEDUPLICATION
    # ====================================================
    logger.info("Step 2: Scanning for EXACT duplicates...")
    
    seen_exact = set()
    after_exact = []
    exact_removed = 0
    
    for record in processed_records:
        norm = record["normalized_text"]
        # A blank normalized string should probably be kept or dropped, but safe to skip dropping if it's unique
        if norm and norm in seen_exact:
            exact_removed += 1
        else:
            seen_exact.add(norm)
            after_exact.append(record)
            
    logger.info(f"-> Removed {exact_removed} EXACT duplicate chunks. Remaining: {len(after_exact)}.")
    
    if not after_exact:
        logger.warning("No records remaining after exact removal!")
        return

    # ====================================================
    # PHASE 2: NEAR-DUPLICATE DEDUPLICATION (TF-IDF + Cosine Similarity)
    # ====================================================
    logger.info("Step 3: Scanning for NEAR duplicates using TF-IDF + Cosine Similarity...")
    
    # 1. Construct vectors over remaining corpus
    texts_to_compare = [r["semantic_text"] for r in after_exact]
    
    # Using word level 1-2 ngrams captures phrase sequences well for similarity
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), 
        min_df=1, 
        stop_words='english',
        lowercase=True
    )
    
    logger.info("Building TF-IDF sparse matrix...")
    tfidf_matrix = vectorizer.fit_transform(texts_to_compare)
    
    logger.info("Computing pairwise cosine similarities...")
    # Fast matrix multiplication
    sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    # 2. Traverse upper triangle of similarity matrix
    to_remove = set()
    near_removed = 0
    
    n = len(after_exact)
    for i in range(n):
        if i in to_remove:
            continue
            
        # Compare against all subsequent documents
        for j in range(i + 1, n):
            if j in to_remove:
                continue
                
            similarity = sim_matrix[i, j]
            if similarity >= near_dup_threshold:
                c1 = after_exact[i]["original_data"]
                # Add semantic text to original dict context so safeguard can check it
                c1["semantic_text"] = after_exact[i]["semantic_text"]
                
                c2 = after_exact[j]["original_data"]
                c2["semantic_text"] = after_exact[j]["semantic_text"]
                
                # Verify Safeguards
                if should_preserve_variant(c1, c2):
                    # One of the safeguards triggered! Retain both.
                    continue
                
                # Truly a duplicate! Remove the second one.
                to_remove.add(j)
                near_removed += 1
                
                # Optional: Sample log for visibility
                if near_removed <= 5:
                    logger.debug(f"[Near-Dup Check] Match found (Sim: {similarity:.3f}) between Chunks {after_exact[i]['index']} and {after_exact[j]['index']}")
                    
    # Compile final set of records
    final_records = []
    for idx, record in enumerate(after_exact):
        if idx not in to_remove:
            # We drop the 'semantic_text' helper key we added during safeguards
            clean_orig = record["original_data"]
            clean_orig.pop("semantic_text", None)
            final_records.append(clean_orig)
            
    logger.info(f"-> Removed {near_removed} NEAR duplicate chunks. Retained {len(final_records)}.")
    
    # ====================================================
    # LOGGING SUMMARY
    # ====================================================
    logger.info("\n" + "="*40 + "\nDEDUPLICATION REPORT\n" + "="*40)
    logger.info(f"Total Chunks Loaded:       {total_loaded}")
    logger.info(f"Exact Duplicates Removed: {exact_removed}")
    logger.info(f"Near Duplicates Removed:  {near_removed}")
    logger.info(f"Final Embedded Count:     {len(final_records)}")
    reduction_percent = ((total_loaded - len(final_records)) / total_loaded) * 100
    logger.info(f"Corpus Compression:       {reduction_percent:.2f}%")
    logger.info("="*40 + "\n")

    # Write results back in original format
    logger.info(f"Saving deduplicated output to: {output_path}...")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(final_records, f, indent=2, ensure_ascii=False)
        
    logger.info("Deduplication successfully completed!")

if __name__ == "__main__":
    INPUT_FILE = os.path.join("data", "processed_semantic_chunks.json")
    OUTPUT_FILE = os.path.join("data", "deduplicated_semantic_chunks.json")
    
    run_deduplication(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        near_dup_threshold=0.88 # Recommended strict overlap boundary
    )
