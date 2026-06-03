from typing import Dict, Any
from langfuse import observe
from app.utils.langfuse_client import langfuse_client
from rag.retrieval import FranchiseRetriever

# Instantiate a single retriever for all tools
retriever = FranchiseRetriever()

@observe(name="tool_execution")
def retrieve(query: str, intent: str = "general") -> Dict[str, Any]:
    """
    Unified RAG tool to get franchise information based on a user query and LLM-classified intent.
    Delegates dynamic retrieval strategy entirely to FranchiseRetriever.
    """
    langfuse_client.update_current_span(
        input={"query": query, "intent": intent, "tool_name": "retrieve"}
    )
    
    chunks = retriever.retrieve(query=query, intent=intent)

    # Guarantee that Chunks 87-92 (the full breakdown table) are included for investment intent
    if intent in ["investment", "fdd_financial"]:
        try:
            if retriever.manager and retriever.manager.vector_store:
                required_ids = {87, 88, 89, 90, 91, 92}
                existing_ids = {c.get("metadata", {}).get("chunk_id") for c in chunks}
                missing_ids = required_ids - existing_ids
                
                if missing_ids:
                    all_docs = retriever.manager.vector_store.docstore._dict.values()
                    for doc in all_docs:
                        if doc.metadata.get("chunk_id") in missing_ids:
                            chunks.insert(0, {
                                "text": doc.page_content,
                                "metadata": doc.metadata,
                                "source_url": doc.metadata.get("url", ""),
                                "rank": 0
                            })
        except Exception as e:
            print("Failed to fetch explicit chunks:", e)
    
    output = {
        "status": "success",
        "retrieved_chunks": chunks
    }
    
    langfuse_client.update_current_span(output=output)
    return output

@observe(name="tool_execution")
def fallback_no_answer() -> Dict[str, Any]:
    """
    Fallback tool for when no suitable answer or tool is found.
    """
    langfuse_client.update_current_span(
        input={"tool_name": "fallback_no_answer"}
    )
    
    output = {
        "status": "fallback",
        "answer": "I do not have that exact information in my current materials, but I encourage you to connect with the WIN franchise team for those specific details."
    }
    langfuse_client.update_current_span(output=output)
    return output
