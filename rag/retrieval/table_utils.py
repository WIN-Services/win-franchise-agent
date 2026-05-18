import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("table_utils")

def detect_table_intent(query: str) -> bool:
    """
    Detects if the user is asking for a full table, complete breakdown, or all rows.
    Supports intermediate words (e.g. "full franchise fee breakdown") and flexible phrasing.
    """
    query_lower = query.lower()
    words = set(re.findall(r'\b\w+\b', query_lower))
    
    # 1. Broad expansion keywords that trigger full tables when combined with table terms
    expansion_triggers = {
        "full", "complete", "show", "itemized", "line", "all", "compare",
        "comparative", "structure", "breakdown", "summarize", "list", "matrix", "every"
    }
    
    # 2. Indicators of table nouns
    table_indicators = {
        "table", "breakdown", "structure", "list", "fee", "fees", "investment", 
        "investments", "pricing", "cost", "costs", "expenditure", "expenditures", 
        "schedule", "matrix", "feature", "features", "service", "services"
    }
    
    # Check for direct phrase matches first (e.g. "all rows", "show all")
    direct_phrases = [
        "all rows", "show all", "itemized details", "line items", "all services", 
        "all features", "fee structure", "pricing structure", "pricing table",
        "comparative structure"
    ]
    for phrase in direct_phrases:
        if phrase in query_lower:
            logger.info(f"Direct phrase table intent detected: '{phrase}'")
            return True
            
    # Check for flexible combinations of trigger words and table indicator words
    has_trigger = any(t in words for t in expansion_triggers)
    has_indicator = any(i in words for i in table_indicators)
    
    if has_trigger and has_indicator:
        logger.info(f"Co-occurrence table intent detected: triggers={expansion_triggers.intersection(words)}, indicators={table_indicators.intersection(words)}")
        return True
        
    return False

class TableIndex:
    """
    Efficient in-memory mapping to retrieve sibling rows of a table without full PKL scans.
    """
    def __init__(self, docstore_dict: Dict[str, Any]):
        # Map table_key -> list of Langchain Documents
        self.table_map = {}
        self._build_index(docstore_dict)

    def _build_index(self, docstore_dict: Dict[str, Any]):
        for doc_id, doc in docstore_dict.items():
            metadata = doc.metadata
            table = metadata.get("table")
            if table:
                # Group by table name and section to avoid collisions
                section = metadata.get("metadata", {}).get("section", "")
                table_key = f"{table}::{section}"
                if table_key not in self.table_map:
                    self.table_map[table_key] = []
                self.table_map[table_key].append(doc)
                
    def get_table_docs(self, table_key: str) -> List[Any]:
        return self.table_map.get(table_key, [])

def reconstruct_table(table_key: str, table_docs: List[Any], max_rows: int = 50) -> Dict[str, Any]:
    """
    Dynamically rebuilds a full Markdown table from sibling documents.
    """
    # Sort docs chronologically by chunk_id
    sorted_docs = sorted(table_docs, key=lambda d: d.metadata.get("chunk_id", 0))
    
    # Deduplicate rows by comparing the non-metadata fields
    unique_rows = []
    seen_hashes = set()
    
    columns_set = set()
    
    ignore_keys = {"chunk_id", "semantic_text", "original_text", "metadata", "table", "table_id", "table_name"}
    
    for doc in sorted_docs:
        # Extract row fields
        row_dict = {k: v for k, v in doc.metadata.items() if k not in ignore_keys}
        row_hash = tuple(sorted(row_dict.items()))
        
        if row_hash not in seen_hashes:
            seen_hashes.add(row_hash)
            unique_rows.append((doc, row_dict))
            columns_set.update(row_dict.keys())
            
            if len(unique_rows) >= max_rows:
                logger.warning(f"Table {table_key} exceeded max_rows {max_rows}. Truncating.")
                break
                
    # Format table to markdown
    # Convert column keys back to Title Case logic similar to pipeline
    def humanize_key(key: str) -> str:
        return " ".join(word.capitalize() for word in key.split("_"))

    # Force a specific ordering for logical presentation if possible, else sort alphabetically
    columns = sorted(list(columns_set))
    
    # Ensure "Description" or "Content" is usually at the end if it exists
    if "description" in columns:
        columns.remove("description")
        columns.append("description")
        
    human_cols = [humanize_key(c) for c in columns]
    
    # Table Header
    parts = table_key.split('::')
    table_name_human = humanize_key(parts[0]) if parts[0] else "Data Table"
    
    md_lines = [
        f"Table: {table_name_human}",
        "|" + "|".join(human_cols) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|"
    ]
    
    for _, row_dict in unique_rows:
        row_vals = []
        for col in columns:
            val = str(row_dict.get(col, "")).replace("\n", " ").replace("|", "\\|")
            row_vals.append(val)
        md_lines.append("|" + "|".join(row_vals) + "|")
        
    full_md_table = "\n".join(md_lines)
    
    # Return Synthesized chunk dict
    base_metadata = unique_rows[0][0].metadata.get("metadata", {})
    return {
        "text": full_md_table,
        "metadata": {
            "section": base_metadata.get("section", table_name_human),
            "table_key": table_key,
            "reconstructed": True,
            "row_count": len(unique_rows)
        }
    }
