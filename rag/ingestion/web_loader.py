import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Any
from .base import BaseLoader

class WebLoader(BaseLoader):
    """Loads and cleans text from a given URL, grouping by headings."""
    
    def __init__(self, url: str):
        self.url = url
        
    def load(self) -> List[Dict[str, Any]]:
        response = requests.get(self.url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove noisy elements
        for element in soup(["script", "style", "nav", "footer", "header", "aside"]):
            element.decompose()
            
        sections = []
        current_section = "Website Intro"
        current_text = []

        for element in soup.body.descendants:
            if getattr(element, 'name', None) in ['h1', 'h2', 'h3']:
                if current_text:
                    text = ' '.join(' '.join(current_text).split())
                    if text:
                        sections.append({
                            "text": text,
                            "metadata": {
                                "source": self.url,
                                "type": "website",
                                "section": current_section
                            }
                        })
                current_section = element.get_text(separator=' ', strip=True)
                current_text = []
            elif isinstance(element, str):
                if element.strip() and element.parent.name not in ['script', 'style']:
                    current_text.append(element.strip())

        if current_text:
            text = ' '.join(' '.join(current_text).split())
            if text:
                sections.append({
                    "text": text,
                    "metadata": {
                        "source": self.url,
                        "type": "website",
                        "section": current_section
                    }
                })
        
        return sections
