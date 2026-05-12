from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseLoader(ABC):
    """Abstract base class for document loaders."""
    
    @abstractmethod
    def load(self) -> List[Dict[str, Any]]:
        """
        Load documents from the source.
        Returns a list of dictionaries containing text and metadata.
        """
        pass
