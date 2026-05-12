from langchain_core.documents import Document
from rag.embeddings import VectorStoreManager
from app.utils.langfuse_client import langfuse_client

def test():
    docs = [
        Document(page_content="WIN Home Inspection is a leading franchise.", metadata={"source": "test"}),
        Document(page_content="The initial franchise fee is $40,000.", metadata={"source": "test"}),
    ]
    
    manager = VectorStoreManager()
    
    # Adding documents should trigger the embedder and log a span in Langfuse
    print("Adding documents...")
    manager.add_documents(docs)
    
    print("Documents added. Vector store saved.")
    
    # Retrieve
    print("Testing retriever...")
    retriever = manager.get_retriever(k=1)
    results = retriever.invoke("What is the initial fee?")
    print(f"Retrieved: {results[0].page_content}")
    
    langfuse_client.flush()

if __name__ == "__main__":
    test()
