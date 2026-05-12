from app.agents import FranchiseAgent
from app.utils.langfuse_client import langfuse_client, create_pipeline_trace

def test():
    # Setup dummy context chunks
    dummy_chunks = [
        {"text": "The initial franchise fee for a WIN Home Inspection franchise is $40,000.", "metadata": {"source": "FDD", "section": "Item 5"}},
        {"text": "WIN Home Inspection offers comprehensive training to all new strategic partners.", "metadata": {"source": "Website"}}
    ]
    
    agent = FranchiseAgent()
    query = "What is the initial franchise fee?"
    
    print("Initializing trace...")
    create_pipeline_trace(query=query)
    
    print(f"Generating response for query: '{query}'...")
    response = agent.generate_response(query=query, context_chunks=dummy_chunks)
    print(f"\nResponse:\n{response}")
    
    # Test hallucination guardrail
    query2 = "Who is the CEO of WIN Home Inspection?"
    print(f"\nGenerating response for query: '{query2}'...")
    response2 = agent.generate_response(query=query2, context_chunks=dummy_chunks)
    print(f"\nResponse:\n{response2}")
    
    langfuse_client.flush()

if __name__ == "__main__":
    test()
