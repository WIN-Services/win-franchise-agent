import re
import logging
from typing import List, Dict, Any

logger = logging.getLogger("table_utils")

def detect_table_intent(query: str) -> bool:
    """
    Detects if the user is asking for a full table, complete breakdown, or all rows.
    Implements a 3-tier heuristic matching system (Direct, Plural Co-occurrence, Helper Modifier).
    """
    query_lower = query.lower()
    words = set(re.findall(r'\b\w+\b', query_lower))
    
    # 1. Direct collection triggers (strongly imply full table)
    collection_triggers = {
        "breakdown", "comparative", "structure", "list", "matrix", "schedule", "table", "itemized"
    }
    
    # 2. General expansion request verbs/adjectives
    general_triggers = {
        "full", "complete", "show", "all", "compare", "summarize", "every", "share", "give", "provide", "get", "tell", "start"
    }
    
    # 3. Plural nouns that imply complete datasets
    plural_indicators = {
        "fees", "costs", "investments", "expenditures", "services", "features", "steps"
    }
    
    # 4. Singular table indicators
    singular_indicators = {
        "fee", "cost", "investment", "expenditure", "service", "feature", "step"
    }
    
    # Heuristic A: Direct mention of a table/breakdown keyword
    if any(t in words for t in collection_triggers):
        logger.info(f"Direct collection keyword table intent detected: {collection_triggers.intersection(words)}")
        return True
        
    # Heuristic B: General trigger co-occurring with plural indicator (e.g. "share the costs")
    has_general = any(t in words for t in general_triggers)
    has_plural = any(i in words for i in plural_indicators)
    if has_general and has_plural:
        logger.info(f"Plural indicator table intent detected: trigger={general_triggers.intersection(words)}, plural={plural_indicators.intersection(words)}")
        return True
        
    # Heuristic C: Helper co-occurring with singular indicator (e.g. "full cost")
    has_helper = any(h in words for h in {"full", "complete", "all", "every", "compare", "itemized"})
    has_singular = any(s in words for s in singular_indicators)
    if has_helper and has_singular:
        logger.info(f"Helper and singular table intent detected: helper={has_helper}, singular={singular_indicators.intersection(words)}")
        return True
        
    # Direct phrase overrides
    direct_phrases = [
        "all rows", "show all", "line items", "pricing table", "fee structure", "pricing structure"
    ]
    for phrase in direct_phrases:
        if phrase in query_lower:
            logger.info(f"Direct override phrase table intent detected: '{phrase}'")
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
