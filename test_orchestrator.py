import sys
from app.orchestrator import Orchestrator

def main():
    orc = Orchestrator()
    query = "What is the breakdown of the investment to start the business?"
    print(f"Testing query: {query}")
    
    intent_route = orc.route(query)
    print(f"Intent routed to Tool Output keys: {intent_route.keys()}")
    
    if "retrieved_chunks" in intent_route:
        for idx, chunk in enumerate(intent_route["retrieved_chunks"]):
            print(f"--- Chunk {idx} ---")
            print(chunk.get("text", "")[:200])

if __name__ == '__main__':
    main()
