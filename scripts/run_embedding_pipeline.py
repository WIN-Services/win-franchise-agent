import os
import sys

# Prevent multiple OpenMP runtime conflict crashes on macOS
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
from dotenv import load_dotenv
import numpy as np
import faiss

# Append project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from rag.embeddings.pipeline import EmbeddingPipeline

# Initialize script logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    # Load Environment variables (.env)
    load_dotenv()
    
    api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
    if not api_key:
        logger.error("CRITICAL: OpenAI API key is missing! Please set it in your .env file.")
        sys.exit(1)
        
    # Configuration Paths
    dedup_path = os.path.join("data", "deduplicated_semantic_chunks.json")
    fallback_path = os.path.join("data", "processed_semantic_chunks.json")
    
    if os.path.exists(dedup_path):
        input_json_path = dedup_path
        logger.info(f"Found DEDUPLICATED corpus. Loading from: {input_json_path}")
    else:
        input_json_path = fallback_path
        logger.info(f"No deduplicated corpus found. Using: {input_json_path}")
        
    output_txt_path = os.path.join("data", "semantic_texts_dump.txt")
    output_vector_dir = settings.VECTOR_DB_PATH or "vector_store"
    
    # 1. Initialize Pipeline
    pipeline = EmbeddingPipeline(
        api_key=api_key,
        model_name=settings.EMBEDDING_MODEL
    )
    
    # 2. Load Raw Chunks
    try:
        raw_chunks = pipeline.load_json(input_json_path)
    except Exception as e:
        logger.error(f"Failed to load input dataset: {e}")
        sys.exit(1)
        
    # 3. Generate Semantic Text & Prepare PKL Data Structure
    logger.info("Step 1: Generating high-quality semantic text formatting for all chunks...")
    logger.info("  → FDD text chunks will be replaced with lightweight stubs.")
    logger.info("  → FDD table chunks (investment, questions) will be kept as-is.")
    
    pkl_records = []
    all_semantic_texts = []
    fdd_stubbed_count = 0
    
    for idx, chunk in enumerate(raw_chunks):
        meta = chunk.get("metadata", {})
        is_fdd = meta.get("source") == "document"
        is_table_chunk = "table" in chunk
        
        # --- FDD Stub Logic ---
        # Replace full FDD narrative text with a short searchable stub.
        # Keep table-type FDD chunks (investment breakdown, question rows) intact.
        working_chunk = chunk
        if is_fdd and not is_table_chunk and chunk.get("text", ""):
            section = meta.get("section", "General FDD Info")
            subsection = meta.get("sub-section", "") or meta.get("subsection", "")
            stub_parts = [f"FDD {section}"]
            if subsection:
                stub_parts.append(f"— {subsection}")
            stub_parts.append(": For detailed information, please connect with the WIN franchise team.")
            stub_text = " ".join(stub_parts)
            working_chunk = {**chunk, "text": stub_text}
            fdd_stubbed_count += 1
        
        semantic_text = pipeline.generate_semantic_text(working_chunk)
        all_semantic_texts.append(semantic_text)
        
        # Merge to match the strict requirements:
        # "Store: chunk_id, original text, semantic_text, metadata, and all fields in raw json"
        record = {
            "chunk_id": idx,
            "semantic_text": semantic_text,
            **working_chunk # Expands "text", "metadata", "table", "row_data" elements naturally
        }
        pkl_records.append(record)
    
    logger.info(f"  → Stubbed {fdd_stubbed_count} FDD narrative chunks. Kept {sum(1 for c in raw_chunks if 'table' in c and c.get('metadata',{}).get('source')=='document')} FDD table chunks intact.")
        
    # Save readable TXT file in data folder as requested
    logger.info(f"Saving full semantic dump text to: {output_txt_path}")
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for i, text in enumerate(all_semantic_texts):
            f.write(f"=== CHUNK {i} ===\n")
            f.write(text)
            f.write("\n\n=====================\n\n")
            
    logger.info("Formatting complete. Generating vector embeddings...")
    
    # 4. Generate Embeddings
    try:
        vectors = pipeline.generate_embeddings_pipeline(all_semantic_texts, batch_size=64)
    except Exception as e:
        logger.error(f"Embedding phase aborted: {e}")
        sys.exit(1)
        
    # 5. Build & Normalize FAISS Index using LangChain wrapper for backend compatibility!
    logger.info("Step 3: Wrapping vectors into LangChain FAISS (DistanceStrategy.COSINE)...")
    
    from langchain_community.vectorstores import FAISS
    from langchain_community.vectorstores.utils import DistanceStrategy
    from rag.embeddings.generator import get_embedder
    
    embedder = get_embedder()
    
    # Prepare (text, vector) pairs
    text_embeddings = list(zip(all_semantic_texts, vectors.tolist()))
    
    # Prepare metadatas matching chunk properties exactly
    lc_metadatas = []
    for idx, chunk in enumerate(raw_chunks):
        raw_meta = chunk.get("metadata", {})
        
        # Promote source_url to a top-level metadata field for easy access
        raw_url = raw_meta.get("url", "")
        if raw_url and raw_url.startswith("http"):
            source_url = raw_url
        else:
            # FDD or doc without a URL — use a descriptive label
            path = raw_meta.get("path", "")
            source_url = f"FDD: {path}" if path else ""
        
        # Store original text & complete dictionary keys so everything is accessible
        raw_meta = chunk.get("metadata", {})
        meta_dict = {
            "chunk_id": idx,
            "semantic_text": all_semantic_texts[idx],
            "original_text": pkl_records[idx].get("text", ""),
            "source_url": source_url,
            **{k: v for k, v in chunk.items() if k != "text"}
        }
        lc_metadatas.append(meta_dict)
        
    # Initialize LangChain FAISS database
    db = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embedder,
        metadatas=lc_metadatas,
        distance_strategy=DistanceStrategy.COSINE
    )
    
    # 6. Save persistent storage artifacts in LangChain's expected format
    logger.info(f"Step 4: Saving LangChain FAISS artifacts inside: '{output_vector_dir}'")
    os.makedirs(output_vector_dir, exist_ok=True)
    db.save_local(folder_path=output_vector_dir)
    logger.info("Artifact storage complete! (index.faiss and index.pkl successfully created)")
    
    # ==========================================
    # SANITY CHECK / TEST BLOCK
    # ==========================================
    logger.info("\n" + "="*40 + "\nSANITY CHECK: Running Test Query Retrieval via LangChain\n" + "="*40)
    
    test_query = "What is the initial franchise fee and start-up investment for a WIN Home Inspection franchise?"
    logger.info(f"Testing query: '{test_query}'")
    
    k = 3
    # Run search using the newly created DB to test lookup
    docs_and_scores = db.similarity_search_with_score(test_query, k=k)
    
    logger.info(f"Top {k} Semantic Matches Found:")
    for rank, (doc, score) in enumerate(docs_and_scores):
        logger.info(f"\n[MATCH #{rank + 1}] (Score/Distance: {score:.4f})")
        logger.info(f"Chunk ID: {doc.metadata.get('chunk_id')}")
        logger.info(f"Section: {doc.metadata.get('metadata', {}).get('section', 'N/A')}")
        logger.info("-" * 20)
        # Display first 250 chars of the semantic text preview
        preview = doc.page_content[:250].replace("\n", " ")
        logger.info(f"Text Preview: {preview}...")
        logger.info("-" * 20)
        
    logger.info("\nEmbedding pipeline script execution successfully completed!")

if __name__ == "__main__":
    main()
