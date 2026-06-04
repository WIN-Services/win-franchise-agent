import os
import sys
import time
import json
import uuid
import boto3
from pathlib import Path

# Add project root to sys.path so we can import from app
sys.path.append(str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from app.main import app
from app.config import settings
from app.utils.sqs_logger import sqs_logger

def run_tests():
    print("======================================================================")
    print("🚀 Starting Integration Test for SQS Chatbot Summary Queue")
    print("======================================================================")

    # 1. Configuration Verification
    print(f"AWS Region:             {settings.AWS_REGION_NAME}")
    print(f"SQS Interaction Queue:  {settings.AWS_SQS_QUEUE_URL}")
    print(f"SQS Summary Queue:      {settings.AWS_SQS_SUMMARY_QUEUE_URL}")
    print(f"AWS Access Key ID:      {settings.AWS_ACCESS_KEY_ID[:8]}... (configured)")

    if not settings.AWS_SQS_SUMMARY_QUEUE_URL:
        print("❌ ERROR: AWS_SQS_SUMMARY_QUEUE_URL is not set!")
        sys.exit(1)

    # 2. Isolated Direct SQS Send Verification
    print("\n----------------------------------------------------------------------")
    print("PART 1: Direct sqs_logger.log_summary_trigger() Call")
    print("----------------------------------------------------------------------")
    
    session_id_part1 = f"e2e-summary-direct-{uuid.uuid4()}"
    print(f"Using Session ID: {session_id_part1}")
    
    try:
        print("Calling log_summary_trigger...")
        # Since it runs synchronously here (without background_tasks wrapping it), we can catch exceptions directly
        sqs_logger.log_summary_trigger(session_id=session_id_part1)
        print("✅ log_summary_trigger executed without errors!")
    except Exception as e:
        print(f"❌ Failed direct log_summary_trigger: {e}")
        sys.exit(1)

    # 3. Full API Chat Route Verification
    print("\n----------------------------------------------------------------------")
    print("PART 2: Testing API `/chat` endpoint with Background Tasks")
    print("----------------------------------------------------------------------")
    
    session_id_part2 = f"e2e-summary-api-{uuid.uuid4()}"
    print(f"Using Session ID: {session_id_part2}")
    
    client = TestClient(app)
    test_query = "Hello, I am testing the summary queue trigger. What services do you offer?"
    print(f"Sending POST /chat with query: '{test_query}'")
    
    response = client.post(
        "/chat",
        json={"query": test_query, "session_id": session_id_part2}
    )
    
    if response.status_code != 200:
        print(f"❌ FastAPI request failed: {response.status_code} - {response.text}")
        sys.exit(1)
        
    res_data = response.json()
    print(f"Received API Response status 200.")
    print(f"- Answer snippet: {res_data.get('answer')[:120]}...")
    print(f"- Session ID: {res_data.get('session_id')}")
    
    # Wait for the background task to complete SQS message sending
    print("Waiting 3 seconds for background tasks to complete...")
    time.sleep(3)
    
    print("\n======================================================================")
    print("🏆 ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("======================================================================")

if __name__ == "__main__":
    run_tests()
