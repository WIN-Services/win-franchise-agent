from fastapi.testclient import TestClient
from app.main import app
import time

client = TestClient(app)

def run_test():
    print("=== Starting Automated Pipeline Test ===")
    session_id = "test-session-" + str(int(time.time()))
    
    # 1. Ask about FDD Initial Fees
    print("\n[PROSPECT]: What is the initial franchise fee?")
    response1 = client.post("/chat", json={"query": "What is the initial franchise fee?", "session_id": session_id})
    data1 = response1.json()
    print(f"[AGENT]: {data1['answer']}")
    print(f"[DEMOGRAPHICS Extracted]: {data1.get('demographics')}")
    
    # Check if agent asks a follow-up question
    if "?" in data1['answer']:
        print("✅ Agent asked an engaging question.")
    else:
        print("❌ Agent did not ask an engaging question.")

    # 2. Provide Demographics naturally
    print("\n[PROSPECT]: That sounds good. By the way, my name is John Doe and my phone number is 555-1234. My zip code is 90210.")
    response2 = client.post("/chat", json={
        "query": "That sounds good. By the way, my name is John Doe and my phone number is 555-1234. My zip code is 90210.",
        "session_id": session_id
    })
    data2 = response2.json()
    print(f"[AGENT]: {data2['answer']}")
    print(f"[DEMOGRAPHICS Extracted]: {data2.get('demographics')}")
    
    demogs = data2.get('demographics', {})
    if demogs.get("name") == "John Doe" and "555-1234" in demogs.get("phone_number", "") and "90210" in demogs.get("pin_code", ""):
        print("✅ Demographics successfully extracted and stored!")
    else:
        print("❌ Demographics missing or incorrect!")

    # 3. Ask about Restrictions (FDD Item 8)
    print("\n[PROSPECT]: Are there any restrictions on sources of products and services?")
    response3 = client.post("/chat", json={
        "query": "Are there any restrictions on sources of products and services?",
        "session_id": session_id
    })
    data3 = response3.json()
    print(f"[AGENT]: {data3['answer']}")
    print(f"[DEMOGRAPHICS Extracted]: {data3.get('demographics')}")
    
    print("\n=== Test Completed ===")

if __name__ == "__main__":
    run_test()
