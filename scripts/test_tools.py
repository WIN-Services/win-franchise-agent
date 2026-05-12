import asyncio
from app.tools import get_franchise_info, get_investment_details, get_process_steps, fallback_no_answer
from app.utils.langfuse_client import langfuse_client, create_pipeline_trace

def test():
    # Wrap in a trace
    create_pipeline_trace(query="Testing tools locally")
    
    print("1. Testing get_franchise_info...")
    res1 = get_franchise_info("What is WIN Home Inspection?")
    print(f"   Chunks retrieved: {len(res1.get('retrieved_chunks', []))}")
    
    print("\n2. Testing get_investment_details...")
    res2 = get_investment_details()
    print(f"   Chunks retrieved: {len(res2.get('retrieved_chunks', []))}")
    
    print("\n3. Testing get_process_steps...")
    res3 = get_process_steps()
    print(f"   Chunks retrieved: {len(res3.get('retrieved_chunks', []))}")
    
    print("\n4. Testing fallback_no_answer...")
    res4 = fallback_no_answer()
    print(f"   Answer: {res4.get('answer')}")
    
    langfuse_client.flush()

if __name__ == "__main__":
    test()
