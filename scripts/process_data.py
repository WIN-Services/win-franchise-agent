import sys
import os
import json

# Add the project root to the python path so we can import from rag
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.ingestion.web_loader import WebLoader
from rag.ingestion.docx_loader import DocxLoader
from rag.chunking.token_chunker import TokenChunker
from rag.embeddings.store import VectorStoreManager
from langchain_core.documents import Document

def main():
    # 1. Ingest Data
    print("Ingesting Website Data...")
    web_loader = WebLoader("https://wini.com/franchise/")
    web_docs = web_loader.load()
    
    print("Ingesting FDD Data...")
    fdd_path = "/Users/aryamantyagi/Downloads/2026 WIN FDD(82160923.5).docx"
    
    try:
        docx_loader = DocxLoader(fdd_path)
        fdd_docs = docx_loader.load()
    except Exception as e:
        print(f"Error loading FDD from {fdd_path}: {e}")
        fdd_docs = []
        
    all_docs = web_docs + fdd_docs
    
    # 2. Chunk Data
    print("Chunking Data...")
    # Using 500 tokens for size and 75 for overlap, fitting in 300-800 size and 50-100 overlap criteria
    chunker = TokenChunker(chunk_size=500, chunk_overlap=75)
    chunked_docs = chunker.chunk(all_docs)
    
    # 3. Save Processed Output
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "processed_documents.json")
    print(f"Saving {len(chunked_docs)} chunks to {output_file}...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(chunked_docs, f, indent=2, ensure_ascii=False)
        
    # 4. Generate Embeddings and Save to Vector Store
    print("Generating embeddings and saving to Vector Store...")
    documents = [
        Document(page_content=chunk["text"], metadata=chunk["metadata"])
        for chunk in chunked_docs
    ]
    manager = VectorStoreManager()
    
    # Optional: Clear existing vector store if you want a fresh index each time
    # (Leaving it out so it updates/creates as intended by the manager)
    manager.add_documents(documents)
    print("Vector Store successfully updated!")
        
    print("Done!")

if __name__ == "__main__":
    main()
