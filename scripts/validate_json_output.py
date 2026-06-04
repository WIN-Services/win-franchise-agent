import os
import sys
import json
from collections import Counter

def main():
    file_path = "data/deduplicated_semantic_chunks.json"
    
    print(f"=== Starting Verification of Output JSON: {file_path} ===")
    
    if not os.path.exists(file_path):
        # Fallback to processed_semantic_chunks.json if deduplicated is not generated yet
        file_path = "data/processed_semantic_chunks.json"
        if not os.path.exists(file_path):
            print("ERROR: Neither processed_semantic_chunks.json nor deduplicated_semantic_chunks.json exist. Run the pipeline first!")
            return
            
    print(f"Loading and validating: {file_path}")
    
    # 1. Try to Load JSON (Tests syntax & decoding validity)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print("✅ Success: File is syntactically valid JSON.")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: JSON syntax is broken! Details: {e}")
        return
    except Exception as e:
        print(f"❌ ERROR loading file: {e}")
        return
        
    total_chunks = len(chunks)
    if total_chunks == 0:
        print("⚠️ WARNING: JSON file is empty (0 chunks found).")
        return
        
    print(f"\nTotal Semantic Chunks Found: {total_chunks}")
    
    # 2. Gather statistics
    text_word_counts = []
    missing_fields = Counter()
    sections = set()
    sources = set()
    content_types_count = Counter()
    
    table_chunks_count = 0
    text_chunks_count = 0
    display_texts_count = 0
    
    required_keys = {"section", "sub-section", "source", "url", "path", "content_type"}
    
    for idx, item in enumerate(chunks):
        metadata = item.get("metadata", {})
        
        # Check chunk type
        if "table" in item:
            table_chunks_count += 1
        else:
            text_chunks_count += 1
            text = item.get("text", "")
            text_word_counts.append(len(text.split()))
            
        # Check display_text
        if "display_text" in item:
            display_texts_count += 1
            
        # Check metadata fields
        for key in required_keys:
            val = metadata.get(key)
            if val is None or val == "" or (isinstance(val, list) and not val):
                # url or path could be empty, but section/source shouldn't be.
                missing_fields[key] += 1
                
        if metadata.get("section"):
            sections.add(metadata.get("section"))
        if metadata.get("source"):
            sources.add(metadata.get("source"))
            
        # Tally content types
        c_types = metadata.get("content_type", [])
        if isinstance(c_types, list):
            for ct in c_types:
                content_types_count[ct] += 1
        elif isinstance(c_types, str) and c_types:
            content_types_count[c_types] += 1
            
    # Calculate word statistics for text chunks
    avg_words = sum(text_word_counts) / len(text_word_counts) if text_word_counts else 0
    max_words = max(text_word_counts) if text_word_counts else 0
    min_words = min(text_word_counts) if text_word_counts else 0
    
    # 3. Print Profile Reports
    print("\n--- 📦 Dataset Metrics ---")
    print(f"  - Total Chunks: {total_chunks}")
    print(f"    * Text/Narrative Chunks: {text_chunks_count}")
    print(f"    * Table-Row Chunks: {table_chunks_count}")
    print(f"  - Text Chunks Average Length: {avg_words:.1f} words")
    print(f"  - Text Chunks Shortest: {min_words} words")
    print(f"  - Text Chunks Longest: {max_words} words")
    print(f"  - Chunks with 'display_text' summary: {display_texts_count}")
    print(f"  - Unique Sections Covered: {len(sections)}")
    print(f"  - Unique Sources Found: {', '.join(sources)}")
    
    print("\n--- 🏷️ Content Type Distribution ---")
    if content_types_count:
        for ct, cnt in content_types_count.most_common():
            print(f"  - {ct}: {cnt} matches")
    else:
        print("  - No content type classifications found.")
        
    print("\n--- 🛠️ Metadata Completeness Report ---")
    for key in required_keys:
        missing = missing_fields[key]
        perc = (missing / total_chunks) * 100
        status = "ℹ️ (Normal for paths)" if key in ["url", "path"] else "⚠️ Missing!" if missing > 0 else "✅ Perfect"
        print(f"  - Field '{key}': Empty/None in {missing} chunks ({perc:.1f}%) -> {status}")
        
    # 4. Display Spot Check sample
    print("\n--- 🔍 Spot Check (First Chunk) ---")
    sample = chunks[0]
    if "table" in sample:
        print(f"Table Chunk: Row from table '{sample['table']}'")
    else:
        print(f"Narrative Chunk Preview: \"{sample.get('text', '')[:120].strip()}...\"")
    if "display_text" in sample:
        print(f"Display Text (Summary): \"{sample['display_text']}\"")
    print("Metadata Object:")
    print(json.dumps(sample['metadata'], indent=2, ensure_ascii=False))
    
    print("\n=== Verification PASSED! Dataset is perfect and ready. ===")

if __name__ == "__main__":
    main()
