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

def load_s3_bucket_name():
    """Load S3_BUCKET_NAME from wini_serverless/.env if possible, fallback to 'win-data-test'"""
    try:
        serverless_env = Path(__file__).parent.parent.parent / "wini_serverless" / ".env"
        if serverless_env.exists():
            with open(serverless_env, "r") as f:
                for line in f:
                    if line.strip().startswith("S3_BUCKET_NAME"):
                        return line.split("=")[1].strip()
    except Exception:
        pass
    return "win-data-test"

def get_s3_client():
    """Initialize boto3 S3 client using project credentials"""
    kwargs = {
        "region_name": settings.AWS_REGION_NAME or "us-east-1"
    }
    if settings.AWS_ACCESS_KEY_ID:
        kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
    if settings.AWS_SECRET_ACCESS_KEY:
        kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY
    return boto3.client("s3", **kwargs)

def poll_s3_for_file(s3_client, bucket, key, timeout=20, interval=2):
    """Poll S3 until the specified key exists or timeout is reached"""
    print(f"Polling S3 bucket '{bucket}' for key '{key}'...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            s3_client.head_object(Bucket=bucket, Key=key)
            print(f"🎉 SUCCESS! Found file in S3 after {time.time() - start_time:.2f} seconds.")
            return True
        except Exception:
            time.sleep(interval)
            print(".", end="", flush=True)
    print("\n❌ Timeout reached. S3 file was not created.")
    return False

def run_e2e_tests():
    print("======================================================================")
    print("🚀 Starting End-to-End Chatbot SQS-S3 Logging Pipeline Test")
    print("======================================================================")

    # 1. Setup AWS Clients and configuration
    bucket_name = load_s3_bucket_name()
    s3_client = get_s3_client()
    
    print(f"AWS Region:     {settings.AWS_REGION_NAME}")
    print(f"SQS Queue URL:  {settings.AWS_SQS_QUEUE_URL}")
    print(f"S3 Bucket:      {bucket_name}")
    print(f"AWS Access Key: {settings.AWS_ACCESS_KEY_ID[:8]}... (configured)")
    
    # -------------------------------------------------------------------------
    # PART 1: Isolated SQS-to-S3 Lambda Trigger Test
    # -------------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("PART 1: Testing SQS -> Lambda -> S3 Trigger Directly")
    print("----------------------------------------------------------------------")
    
    session_id_part1 = f"e2e-test-direct-{uuid.uuid4()}"
    print(f"Using Session ID: {session_id_part1}")
    
    sqs_client = boto3.client(
        "sqs",
        region_name=settings.AWS_REGION_NAME,
        aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
        aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
    )
    
    direct_payload = {
        "session_id": session_id_part1,
        "message_index": 1,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "query": "Direct SQS trigger test message",
        "response": {
            "answer": "This is a direct test message to verify the Lambda function is triggered by SQS and writes to S3.",
            "sources": [],
            "demographics": {},
            "metadata": {
                "history_length": 1,
                "retrieved_chunks": []
            }
        }
    }
    
    # Define S3 location for this session
    expected_s3_key_part1 = f"franchise_chatbot/conversation/{session_id_part1}/000001.json"
    
    print("Sending message to SQS...")
    try:
        # Prepare send message params
        params = {
            "QueueUrl": settings.AWS_SQS_QUEUE_URL,
            "MessageBody": json.dumps(direct_payload)
        }
        # If it's a FIFO queue, we must supply MessageGroupId (and MessageDeduplicationId if content-based deduplication is off)
        if ".fifo" in settings.AWS_SQS_QUEUE_URL:
            print("Detected FIFO queue. Supplying MessageGroupId...")
            params["MessageGroupId"] = session_id_part1
            params["MessageDeduplicationId"] = session_id_part1
            
        send_resp = sqs_client.send_message(**params)
        print(f"SQS Message sent successfully! MessageId: {send_resp.get('MessageId')}")
    except Exception as e:
        print(f"❌ Failed to send direct message to SQS: {e}")
        sys.exit(1)
        
    # Poll S3 for the created file
    if poll_s3_for_file(s3_client, bucket_name, expected_s3_key_part1):
        # Fetch and verify the file content
        s3_obj = s3_client.get_object(Bucket=bucket_name, Key=expected_s3_key_part1)
        stored_payload = json.loads(s3_obj["Body"].read().decode("utf-8"))
        print("\nStored S3 Content:")
        print(json.dumps(stored_payload, indent=2))
        print("✅ Part 1 complete: SQS -> Lambda -> S3 works perfectly!")
    else:
        print("❌ Part 1 failed: File was not created in S3.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # PART 2: Full API-to-SQS-to-S3 Pipeline Test
    # -------------------------------------------------------------------------
    print("\n----------------------------------------------------------------------")
    print("PART 2: Testing Full API -> SQS -> Lambda -> S3 Pipeline")
    print("----------------------------------------------------------------------")
    
    session_id_part2 = f"e2e-test-api-{uuid.uuid4()}"
    print(f"Using Session ID: {session_id_part2}")
    
    client = TestClient(app)
    test_query = "What is the initial investment and franchise fee for WIN Home Inspection?"
    print(f"Sending POST /chat with query: '{test_query}'")
    
    # We catch any SQS sending errors during the API request
    response = client.post(
        "/chat",
        json={"query": test_query, "session_id": session_id_part2}
    )
    
    if response.status_code != 200:
        print(f"❌ FastAPI request failed: {response.status_code} - {response.text}")
        sys.exit(1)
        
    res_data = response.json()
    print(f"Received API Response:\n- Answer: {res_data.get('answer')[:120]}...")
    print(f"- Demographics: {res_data.get('demographics')}")
    print(f"- History length: {res_data.get('history_length')}")
    
    # Wait for the background task to complete SQS message sending
    print("Waiting 3 seconds for FastAPI background tasks to execute SQS sending...")
    time.sleep(3)
    
    # The Lambda should be triggered by SQS.
    # Note that in SQS logger, the message is sent. Since the message didn't have message_index,
    # the lambda fallback will call get_next_sequence_number and save it as 000001.json.
    expected_s3_key_part2 = f"franchise_chatbot/conversation/{session_id_part2}/000001.json"
    
    # Poll S3 for the created file
    if poll_s3_for_file(s3_client, bucket_name, expected_s3_key_part2):
        s3_obj = s3_client.get_object(Bucket=bucket_name, Key=expected_s3_key_part2)
        stored_payload = json.loads(s3_obj["Body"].read().decode("utf-8"))
        print("\nStored S3 Content from API interaction:")
        print(json.dumps(stored_payload, indent=2))
        print("✅ Part 2 complete: Full API -> SQS -> Lambda -> S3 pipeline works perfectly!")
    else:
        print("❌ Part 2 failed: API logged interaction did not result in an S3 file.")
        sys.exit(1)
        
    print("\n======================================================================")
    print("🏆 ALL TESTS PASSED SUCCESSFULLY! THE LOGGING PIPELINE IS FULLY OPERATIONAL!")
    print("======================================================================")

if __name__ == "__main__":
    run_e2e_tests()
