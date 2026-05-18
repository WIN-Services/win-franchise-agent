from typing import Dict, Any
from langfuse import observe
from app.utils.langfuse_client import langfuse_client
from rag.retrieval import FranchiseRetriever

# Instantiate a single retriever for all tools
retriever = FranchiseRetriever()

@observe(name="tool_execution")
def get_franchise_info(query: str) -> Dict[str, Any]:
    """
    Main RAG tool to get franchise information based on a user query.
    """
    langfuse_client.update_current_span(
        input={"query": query, "tool_name": "get_franchise_info"}
    )
    
    chunks = retriever.retrieve(query)
    
    output = {
        "status": "success",
        "retrieved_chunks": chunks
    }
    
    langfuse_client.update_current_span(output=output)
    return output

@observe(name="tool_execution")
def get_investment_details(query: str = None) -> Dict[str, Any]:
    """
    Retrieves specific information regarding investment details and initial franchise fees.
    """
    search_query = query if query else "What are the investment details, initial franchise fee, and costs?"
    langfuse_client.update_current_span(
        input={"query": search_query, "tool_name": "get_investment_details"}
    )
    
    chunks = retriever.retrieve(search_query)
    
    output = {
        "status": "success",
        "retrieved_chunks": chunks
    }
    
    langfuse_client.update_current_span(output=output)
    return output

@observe(name="tool_execution")
def get_process_steps(query: str = None) -> Dict[str, Any]:
    """
    Retrieves the steps and timeline required to become a franchise owner.
    """
    search_query = query if query else "What are the steps and process timeline to become a franchise owner?"
    langfuse_client.update_current_span(
        input={"query": search_query, "tool_name": "get_process_steps"}
    )
    
    chunks = retriever.retrieve(search_query)
    
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
        "status": "success",
        "answer": "I'm sorry, I couldn't find the exact information you're looking for in my knowledge base. Would you like to speak with a franchise representative?"
    }
    
    langfuse_client.update_current_span(output=output)
    return output
