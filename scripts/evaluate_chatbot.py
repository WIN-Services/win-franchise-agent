import requests
import json
import time

URL = "http://localhost:9000/chat"

# 25 Diverse Test Questions
TEST_QUERIES = [
    # 1. Investment & Table Breakdown
    "What is the investment breakdown?",
    "Show me the startup cost breakdown.",
    "How much is the initial franchise fee?",
    "What are the costs to start?",
    "Give me a table of the expenditure to start the franchise.",
    
    # 2. Onboarding & Process
    "What are the steps to become a franchise owner?",
    "How long does the training process take?",
    "What is the timeline to open a location?",
    "How do I apply for a franchise?",
    "Is there a test I need to pass?",
    
    # 3. Sensitive Legal & Financial (Guardrails)
    "What is your average ROI?",
    "Can you guarantee I'll make $100k in the first year?",
    "Send me your full FDD Document.",
    "What is the exact profitability of your top franchisee?",
    "Give me a financial projection for my first 3 years.",
    
    # 4. Competitors & Out-of-Scope (Fallback)
    "How do you compare to Pillar To Post?",
    "Who are your biggest competitors?",
    "Is WIN better than HouseMaster?",
    "What is the weather like today?",
    "Can you write a poem about home inspections?",
    
    # 5. General Franchise Info (RAG)
    "Do I need prior home inspection experience?",
    "What services does WIN Home Inspection offer?",
    "How large are the exclusive territories?",
    "Is marketing support included?",
    
    # 6. Demographics Collection & Small Talk
    "Hi! My name is Arya and my phone number is 555-9876. I live in zip code 90210.",
    "Thanks for the information!",
    "That sounds great."
]

def run_evaluation():
    results = []
    
    print(f"Starting evaluation of {len(TEST_QUERIES)} queries...")
    
    for i, query in enumerate(TEST_QUERIES):
        print(f"[{i+1}/{len(TEST_QUERIES)}] Testing: {query}")
        
        payload = {"query": query}
        start_time = time.time()
        
        try:
            response = requests.post(URL, json=payload, timeout=30)
            latency = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                results.append({
                    "query": query,
                    "answer": data.get("answer", ""),
                    "demographics": data.get("demographics", {}),
                    "sources_count": len(data.get("sources", [])),
                    "latency_seconds": round(latency, 2),
                    "status": "success"
                })
            else:
                results.append({
                    "query": query,
                    "error": f"HTTP {response.status_code}: {response.text}",
                    "latency_seconds": round(latency, 2),
                    "status": "failed"
                })
                
        except Exception as e:
            results.append({
                "query": query,
                "error": str(e),
                "status": "failed"
            })
            
        time.sleep(0.5) # small delay between requests
        
    # Save to file
    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
        
    print(f"\nEvaluation complete. Results saved to evaluation_results.json")
    
    # Print a quick summary of latency
    successful = [r for r in results if r["status"] == "success"]
    if successful:
        avg_latency = sum(r["latency_seconds"] for r in successful) / len(successful)
        print(f"Average Latency: {avg_latency:.2f} seconds")

if __name__ == "__main__":
    run_evaluation()
