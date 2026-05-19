import os
import sys
import re
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add project root to PYTHONPATH so we can import app and rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from rag.chunking.semantic_chunker import SemanticChunker

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def parse_blocks(file_path: str):
    """
    Parses the knowledge base text file structured into metadata and content blocks separated by '---'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Knowledge base file not found at {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    blocks = []
    current_meta = {}
    current_content = []
    in_meta = False
    
    lines = content.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            if not in_meta:
                # If we were accumulating content, save the completed block
                if current_content or current_meta:
                    blocks.append({
                        "metadata": current_meta,
                        "content": "\n".join(current_content).strip()
                    })
                current_meta = {}
                current_content = []
                in_meta = True
            else:
                in_meta = False
            continue
            
        if in_meta:
            if ":" in line:
                key, val = line.split(":", 1)
                current_meta[key.strip().lower()] = val.strip()
        else:
            current_content.append(line)
            
    # Capture trailing block
    if current_content or current_meta:
        blocks.append({
            "metadata": current_meta,
            "content": "\n".join(current_content).strip()
        })
        
    # Filter empty blocks
    valid_blocks = []
    for b in blocks:
        # A valid block has text content or metadata
        if b["content"] or b["metadata"]:
            valid_blocks.append(b)
            
    logger.info(f"Successfully parsed {len(valid_blocks)} blocks from {file_path}")
    return valid_blocks


def slugify_header(header_str: str) -> str:
    """
    Converts a header string to clean snake_case keys.
    """
    # Remove accents/normalize
    import unicodedata
    norm = unicodedata.normalize("NFKC", header_str)
    slug = norm.lower()
    # Replace non-alphanumeric (excluding underscores) with underscore
    slug = re.sub(r'[^a-z0-9_]+', '_', slug)
    # Collapse repeated underscores and strip outer ones
    slug = re.sub(r'_+', '_', slug).strip('_')
    return slug

def get_clean_table_name(first_header: str) -> str:
    """
    Derives a clean 'table' type classifier based on the first column name.
    """
    slug = slugify_header(first_header)
    if "expenditure" in slug:
        return "expenditure"
    if "fee" in slug:
        return "fee"
    # Remove common prefix words like "type_of_"
    slug = re.sub(r'^(type_of_|list_of_|table_of_|summary_of_)', '', slug)
    return slug

def process_table_block(table_lines: List[str], chunker) -> List[Dict[str, Any]]:
    """
    Parses Markdown table lines into a list of row objects.
    """
    if len(table_lines) < 3:
        return []
        
    header_row = table_lines[0]
    
    # Extract headers (exclude outer boundaries if present)
    headers = [h.strip() for h in header_row.split('|')]
    if headers[0] == "": headers.pop(0)
    if headers and headers[-1] == "": headers.pop()
    
    keys = [slugify_header(h) for h in headers]
    table_name = get_clean_table_name(headers[0]) if headers else "unknown_table"
    
    records = []
    
    # Data rows start at index 2 (index 1 is the --- divider)
    for row_line in table_lines[2:]:
        if not row_line.strip():
            continue
            
        cols = [c.strip() for c in row_line.split('|')]
        if cols[0] == "": cols.pop(0)
        if cols and cols[-1] == "": cols.pop()
        
        row_data = {}
        for idx, key in enumerate(keys):
            raw_val = cols[idx] if idx < len(cols) else ""
            # Clean unicode characters inside values using semantic chunker utility
            cleaned_val = chunker._clean_text(raw_val)
            row_data[key] = cleaned_val
            
        records.append({
            "type": "table_row",
            "table": table_name,
            "row_data": row_data
        })
        
    return records

def parse_tables_and_text(text: str, chunker) -> List[Dict[str, Any]]:
    """
    Segments text into non-table blocks and structured markdown table rows.
    """
    lines = text.split("\n")
    segments = []
    current_text_lines = []
    
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Detect table start: current line looks like header row, 
        # next line is the markdown separator.
        if (
            stripped.startswith('|') and 
            stripped.endswith('|') and 
            i + 1 < len(lines) and 
            re.match(r'^\s*\|[\s\-:|]+\|\s*$', lines[i+1])
        ):
            # Flush accumulated text first
            if current_text_lines:
                segments.append({
                    "type": "text",
                    "content": "\n".join(current_text_lines).strip()
                })
                current_text_lines = []
                
            # Start consuming the table block
            table_lines = [line, lines[i+1]]
            i += 2
            
            # Consume contiguous row lines
            while i < len(lines) and lines[i].strip().startswith('|'):
                table_lines.append(lines[i])
                i += 1
                
            # Parse the table block and append row segments
            table_segments = process_table_block(table_lines, chunker)
            segments.extend(table_segments)
            continue
        else:
            current_text_lines.append(line)
            i += 1
            
    # Flush remaining text
    if current_text_lines:
        segments.append({
            "type": "text",
            "content": "\n".join(current_text_lines).strip()
        })
        
    return segments

def main():
    # Make sure environment variables are loaded
    load_dotenv()
    
    input_file = "data/extracted_knowledge_base.txt"
    output_file = "data/processed_semantic_chunks.json"
    
    logger.info(f"Starting Semantic Chunking Pipeline...")
    logger.info(f"Using configuration -> Overlap: {settings.CHUNK_OVERLAP} tokens, Model: {settings.EMBEDDING_MODEL}")
    
    # 1. Parse knowledge base blocks
    try:
        blocks = parse_blocks(input_file)
    except Exception as e:
        logger.error(f"Failed to parse input file: {e}")
        return
        
    # 2. Initialize Semantic Chunker
    chunker = SemanticChunker(
        overlap_tokens=settings.CHUNK_OVERLAP,
        breakpoint_threshold_percentile=92 
    )
    
    all_chunks_output = []
    
    # 3. Process each block
    for i, block in enumerate(blocks):
        content = block["content"]
        raw_meta = block["metadata"]
        
        if not content:
            continue
            
        logger.info(f"[{i+1}/{len(blocks)}] Ingesting section: '{raw_meta.get('section', 'Unknown')}'...")
        
        # Format url_or_path metadata properly
        url_or_path = raw_meta.get("url_or_path", "")
        url = url_or_path if url_or_path.startswith("http") else ""
        path = "" if url_or_path.startswith("http") else url_or_path
        
        # Segment into tables vs standard narrative text
        segments = parse_tables_and_text(content, chunker)
        
        for segment in segments:
            if segment["type"] == "table_row":
                # Build structured table record mapping row data to TOP level keys
                row_data = segment["row_data"]
                chunk_record = {
                    "metadata": {
                        "section": raw_meta.get("section", ""),
                        "sub-section": raw_meta.get("subsection", ""),
                        "source": raw_meta.get("source", ""),
                        "url": url,
                        "path": path
                    },
                    "table": segment["table"],
                    **row_data
                }
                all_chunks_output.append(chunk_record)
                
            elif segment["type"] == "text":
                text_content = segment["content"]
                if not text_content:
                    continue
                    
                # Perform semantic chunking on pure narrative text
                chunks = chunker.chunk_text(text_content)
                
                # Structure into traditional text chunks
                for idx, chunk_text in enumerate(chunks):
                    chunk_record = {
                        "text": chunk_text,
                        "metadata": {
                            "section": raw_meta.get("section", ""),
                            "sub-section": raw_meta.get("subsection", ""),
                            "source": raw_meta.get("source", ""),
                            "url": url,
                            "path": path,
                            "chunk_index_in_section": idx
                        }
                    }
                    all_chunks_output.append(chunk_record)
            
    logger.info(f"Chunking complete! Generated {len(all_chunks_output)} total records.")
    
    # 4. Write output to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks_output, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully wrote output json to: {output_file}")

if __name__ == "__main__":
    main()
