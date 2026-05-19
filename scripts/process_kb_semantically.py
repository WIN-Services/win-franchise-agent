import os
import sys
import re
import json
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

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

_summarizer_llm = None

def get_summary(text_val: str) -> str:
    """
    Generates a compressed summary of the text value, reducing content size by approximately 50%.
    """
    global _summarizer_llm
    if not text_val.strip():
        return ""
    if _summarizer_llm is None:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if not api_key:
            logger.warning("No API key found for summarization, falling back to truncation.")
            return text_val[:len(text_val)//2] + "..."
        _summarizer_llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0,
            max_tokens=250
        )
    try:
        messages = [
            SystemMessage(content="You are a professional assistant. Your task is to compress the provided text. Reduce the content size by approximately 50% while retaining all specific details, core facts, metrics, and structural layout. Preserve key information and do not generalize excessively. Output only the compressed text, with no introduction or extra words."),
            HumanMessage(content=text_val)
        ]
        res = _summarizer_llm.invoke(messages)
        return res.content.strip()
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        return text_val[:len(text_val)//2] + "..."

def classify_content_type(text: str) -> List[str]:
    """
    Classifies content semantically into all matching designated categories based on keyword mappings.
    """
    if not text:
        return []
    text_lower = text.lower()
    matched = []
    
    # pricing
    pricing_kws = ["pricing", "cost", "investment", "fee", "expense", "royalty", "royalties", "ad fee", "financial", "$", "expenditure"]
    if any(k in text_lower for k in pricing_kws):
        matched.append("pricing")
        
    # steps
    steps_kws = ["step 1", "step 2", "step 3", "step 4", "step 5", "steps", "process", "onboarding", "how to join", "flow", "sequence", "phase"]
    if any(k in text_lower for k in steps_kws):
        matched.append("steps")
        
    # training
    training_kws = ["training", "train", "certify", "certification", "in-house training", "academy", "education", "course", "bootcamp"]
    if any(k in text_lower for k in training_kws):
        matched.append("training")
        
    # support
    support_kws = ["support", "assist", "help", "coach", "guide", "hotline", "team", "advisor", "ongoing support"]
    if any(k in text_lower for k in support_kws):
        matched.append("support")
        
    # technology
    tech_kws = ["technology", "software", "hardware", "tool", "app", "platform", "portal", "dashboard", "tablet", "device", "proprietary"]
    if any(k in text_lower for k in tech_kws):
        matched.append("technology")
        
    # marketing
    marketing_kws = ["marketing", "promote", "advertising", "campaign", "lead", "brand", "social media", "seo", "local market", "collateral", "flyer"]
    if any(k in text_lower for k in marketing_kws):
        matched.append("marketing")
        
    # territory
    territory_kws = ["territory", "location", "area", "exclusive", "zip code", "market", "demographic", "population", "region"]
    if any(k in text_lower for k in territory_kws):
        matched.append("territory")
        
    # faq
    faq_kws = ["faq", "frequently asked", "question", "q&a", "q:", "a:"]
    if any(k in text_lower for k in faq_kws):
        matched.append("faq")
        
    # legal
    legal_kws = ["legal", "disclosure", "fdd", "agreement", "contract", "item 1", "item 19", "compliance", "regulation"]
    if any(k in text_lower for k in legal_kws):
        matched.append("legal")
        
    # requirements
    req_kws = ["require", "qualification", "criteria", "background", "eligible", "minimum", "experience", "license", "mandatory", "prerequisite"]
    if any(k in text_lower for k in req_kws):
        matched.append("requirements")
        
    # benefits
    benefits_kws = ["benefit", "advantage", "why win", "freedom", "flexibility", "growth", "potential", "pro", "perk", "treat you like family", "successful"]
    if any(k in text_lower for k in benefits_kws):
        matched.append("benefits")
        
    return matched

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
            source_type = raw_meta.get("source", "").strip().lower()
            section_name = raw_meta.get("section", "").strip()
            sub_section_name = (raw_meta.get("subsection", "") or raw_meta.get("sub-section", "")).strip()

            if segment["type"] == "table_row":
                # Build structured table record mapping row data to TOP level keys
                row_data = segment["row_data"]
                
                # Format row data as a string for classification and summarization
                row_text_representation = ", ".join(f"{k.replace('_', ' ').capitalize()}: {v}" for k, v in row_data.items() if k not in ("table", "metadata"))
                text_to_classify = f"{section_name} {sub_section_name} {row_text_representation}"
                content_type = classify_content_type(text_to_classify)
                
                chunk_record = {
                    "metadata": {
                        "section": section_name,
                        "sub-section": sub_section_name,
                        "source": source_type,
                        "url": url,
                        "path": path,
                        "content_type": content_type
                    },
                    "table": segment["table"],
                    **row_data
                }
                
                # For document source, generate summary display_text
                if source_type == "document":
                    logger.info(f"Generating display_text summary for document table row chunk...")
                    chunk_record["display_text"] = get_summary(row_text_representation)
                    
                all_chunks_output.append(chunk_record)
                
            elif segment["type"] == "text":
                text_content = segment["content"]
                if not text_content:
                    continue
                    
                # Perform semantic chunking on pure narrative text
                chunks = chunker.chunk_text(text_content)
                
                # Structure into traditional text chunks
                for idx, chunk_text in enumerate(chunks):
                    text_to_classify = f"{section_name} {sub_section_name} {chunk_text}"
                    content_type = classify_content_type(text_to_classify)
                    
                    chunk_record = {
                        "text": chunk_text,
                        "metadata": {
                            "section": section_name,
                            "sub-section": sub_section_name,
                            "source": source_type,
                            "url": url,
                            "path": path,
                            "content_type": content_type,
                            "chunk_index_in_section": idx
                        }
                    }
                    
                    # For document source, generate summary display_text
                    if source_type == "document":
                        logger.info(f"Generating display_text summary for document text chunk...")
                        chunk_record["display_text"] = get_summary(chunk_text)
                        
                    all_chunks_output.append(chunk_record)
            
    logger.info(f"Chunking complete! Generated {len(all_chunks_output)} total records.")
    
    # 4. Write output to JSON
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks_output, f, indent=2, ensure_ascii=False)
        
    logger.info(f"Successfully wrote output json to: {output_file}")

if __name__ == "__main__":
    main()
