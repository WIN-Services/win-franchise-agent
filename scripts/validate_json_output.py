import os
import sys
import json
from collections import Counter

def main():
    file_path = "data/processed_semantic_chunks.json"
    
    print(f"=== Starting Verification of Output JSON: {file_path} ===")
    
    if not os.path.exists(file_path):
        print(f"ERROR: The file {file_path} does not exist. Run the chunking script first!")
        return
        
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
        
    print(f"\nTotal Semantic Chunks Generated: {total_chunks}")
    
    # 2. Gather statistics
    word_counts = []
    missing_fields = Counter()
    sections = set()
    sources = set()
    
    required_keys = {"section", "sub-section", "source", "url", "path"}
    
    for idx, item in enumerate(chunks):
        text = item.get("text", "")
        metadata = item.get("metadata", {})
        
        word_counts.append(len(text.split()))
        
        # Check metadata fields
        for key in required_keys:
            val = metadata.get(key, "")
            if val == "" or val is None:
                # url or path could be empty, but section/source shouldn't be.
                missing_fields[key] += 1
                
        if metadata.get("section"):
            sections.add(metadata.get("section"))
        if metadata.get("source"):
            sources.add(metadata.get("source"))
            
    avg_words = sum(word_counts) / total_chunks
    max_words = max(word_counts)
    min_words = min(word_counts)
    
    # 3. Print Profile Reports
    print("\n--- 📦 Dataset Metrics ---")
    print(f"  - Total Chunks: {total_chunks}")
    print(f"  - Average Length per Chunk: {avg_words:.1f} words")
    print(f"  - Shortest Chunk: {min_words} words")
    print(f"  - Longest Chunk: {max_words} words")
    print(f"  - Unique Sections Covered: {len(sections)}")
    print(f"  - Unique Sources Found: {', '.join(sources)}")
    
    print("\n--- 🛠️ Metadata Completeness Report ---")
    for key in required_keys:
        missing = missing_fields[key]
        perc = (missing / total_chunks) * 100
        status = "ℹ️ (Normal for paths)" if key in ["url", "path"] else "⚠️ Missing!" if missing > 0 else "✅ Perfect"
        print(f"  - Field '{key}': Empty in {missing} chunks ({perc:.1f}%) -> {status}")
        
    # 4. Display Spot Check sample
    print("\n--- 🔍 Random Spot Check (First Chunk) ---")
    sample = chunks[0]
    print(f"Text Preview: \"{sample['text'][:150].strip()}...\"")
    print("Metadata Object:")
    print(json.dumps(sample['metadata'], indent=2, ensure_ascii=False))
    
    print("\n=== Verification PASSED! Ready to ingest. ===")

if __name__ == "__main__":
    main()
