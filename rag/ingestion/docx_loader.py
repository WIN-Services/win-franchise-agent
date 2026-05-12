from typing import Dict, List, Any
import docx
import os
from .base import BaseLoader

class DocxLoader(BaseLoader):
    """Loads text from a DOCX file, grouping by ITEM sections."""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        
    def load(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
            
        doc = docx.Document(self.file_path)
        
        sections = []
        current_section = "General FDD Info"
        current_text = []
        
        for p in doc.paragraphs:
            text = p.text.strip()
            if not text:
                continue
                
            # FDD specific heuristic for headers
            if text.startswith("ITEM ") and "\t" not in text and len(text) < 150:
                if current_text:
                    sections.append({
                        "text": "\n\n".join(current_text),
                        "metadata": {
                            "source": self.file_path,
                            "type": "fdd",
                            "section": current_section
                        }
                    })
                current_section = text
                current_text = []
            else:
                current_text.append(text)
                
        # Handle the tables by appending them to the current text
        # (A simplified approach: dump table contents at the end or interleave if we had paragraph ordering)
        # We will just append them at the end of whatever section they are in, or general text
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join([cell.text.strip() for cell in row.cells if cell.text.strip()])
                if row_text:
                    current_text.append(row_text)

        if current_text:
            sections.append({
                "text": "\n\n".join(current_text),
                "metadata": {
                    "source": self.file_path,
                    "type": "fdd",
                    "section": current_section
                }
            })
            
        return sections
