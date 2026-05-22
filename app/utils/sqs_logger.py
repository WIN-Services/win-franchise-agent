import logging
import boto3
import json
import datetime
from app.config import settings

logger = logging.getLogger("sqs_logger")

class SQSLogger:
    """
    Utility class to log chatbot query and response interactions to AWS SQS.
    Uses lazy-initialization for the boto3 SQS client and degrades gracefully
    if SQS is not configured or fails.
    """

    def __init__(self):
        self._sqs_client = None
        self._initialized = False

    def _init_client(self) -> bool:
        """
        Initializes the boto3 SQS client if not already done.
        Returns True if the client is successfully ready, False otherwise.
        """
        if self._initialized:
            return True

        if not settings.AWS_SQS_QUEUE_URL and not settings.AWS_SQS_SUMMARY_QUEUE_URL:
            # Silent warning to prevent log pollution if intentionally disabled
            logger.warning("Neither AWS_SQS_QUEUE_URL nor AWS_SQS_SUMMARY_QUEUE_URL is set. AWS SQS logging is disabled.")
            return False

        try:
            # Build boto3 client configuration
            kwargs = {
                "region_name": settings.AWS_REGION_NAME or "us-east-1"
            }
            # Use explicit credentials if configured, otherwise fall back to standard boto3 credential resolution
            if settings.AWS_ACCESS_KEY_ID:
                kwargs["aws_access_key_id"] = settings.AWS_ACCESS_KEY_ID
            if settings.AWS_SECRET_ACCESS_KEY:
                kwargs["aws_secret_access_key"] = settings.AWS_SECRET_ACCESS_KEY

            self._sqs_client = boto3.client("sqs", **kwargs)
            self._initialized = True
            logger.info("Successfully initialized SQS logging client.")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SQS client: {e}", exc_info=True)
            return False

    def log_interaction(
        self,
        session_id: str,
        query: str,
        response_answer: str,
        demographics: dict,
        sources: list,
        history_length: int,
        chunk_metadata: list = None
    ) -> None:
        """
        Constructs a structured interaction payload and sends it to the SQS queue.
        This is intended to run as an asynchronous background task.
        """
        if not self._init_client():
            return

        try:
            # Incremental message/turn count. Turn 1 contains 2 messages, so turn = history_length // 2
            message_id = max(1, history_length // 2)
            timestamp = datetime.datetime.utcnow().isoformat() + "Z"

            payload = {
                "session_id": session_id,
                "message_id": message_id,
                "timestamp": timestamp,
                "query": query,
                "response": {
                    "answer": response_answer,
                    "sources": sources,
                    "demographics": demographics,
                    "metadata": {
                        "history_length": history_length,
                        "retrieved_chunks": chunk_metadata or []
                    }
                }
            }

            send_params = {
                "QueueUrl": settings.AWS_SQS_QUEUE_URL,
                "MessageBody": json.dumps(payload)
            }
            
            # FIFO queues require MessageGroupId and MessageDeduplicationId (since ContentBasedDeduplication=false)
            if ".fifo" in settings.AWS_SQS_QUEUE_URL:
                send_params["MessageGroupId"] = session_id
                send_params["MessageDeduplicationId"] = f"{session_id}-{message_id}-{timestamp}"

            self._sqs_client.send_message(**send_params)
            logger.info(f"SQS log message sent successfully. session_id={session_id}, message_id={message_id}")
        except Exception as e:
            logger.error(f"Failed to send interaction log to SQS queue: {e}", exc_info=True)

    def log_summary_trigger(self, session_id: str) -> None:
        """
        Constructs a minimal summary trigger payload and sends it to the summary SQS queue.
        This is intended to run as an asynchronous background task.
        """
        if not self._init_client():
            return

        if not settings.AWS_SQS_SUMMARY_QUEUE_URL:
            logger.warning("AWS_SQS_SUMMARY_QUEUE_URL is not set. Summary trigger is disabled.")
            return

        try:
            timestamp = datetime.datetime.utcnow().isoformat() + "Z"
            payload = {
                "conversation_id": session_id,
                "timestamp": timestamp
            }

            send_params = {
                "QueueUrl": settings.AWS_SQS_SUMMARY_QUEUE_URL,
                "MessageBody": json.dumps(payload)
            }
            
            # FIFO queues require MessageGroupId and MessageDeduplicationId
            if ".fifo" in settings.AWS_SQS_SUMMARY_QUEUE_URL:
                send_params["MessageGroupId"] = session_id
                send_params["MessageDeduplicationId"] = f"{session_id}-{timestamp}"

            self._sqs_client.send_message(**send_params)
            logger.info(f"SQS summary trigger message sent successfully. session_id={session_id}")
        except Exception as e:
            logger.error(f"Failed to send summary trigger to SQS queue: {e}", exc_info=True)


# Module-level singleton
sqs_logger = SQSLogger()
