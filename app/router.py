from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Optional
import uuid

from app.utils.langfuse_client import langfuse_client
from app.orchestrator import Orchestrator
from app.conversation import conversation_manager
from app.utils.sqs_logger import sqs_logger

router = APIRouter()

# Singleton orchestrator
_orchestrator = Orchestrator()


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None   # Client sends this to maintain conversation

class Source(BaseModel):
    title: str
    url: Optional[str] = None
    section: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Source]
    session_id: str                    # Always echoed back so client can continue the conversation
    history_length: int                # How many messages are stored for this session
    demographics: Optional[dict] = None
    persona: str = "exploring"


# ---------------------------------------------------------------------------
# POST /chat
# ---------------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, background_tasks: BackgroundTasks):
    """
    Franchise Chatbot entry point.
    """
    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Assign or accept a session ID
    session_id = request.session_id or str(uuid.uuid4())

    # Retrieve existing conversation history, session state, and demographics for this session
    history = conversation_manager.get_history(session_id)
    session_state = conversation_manager.get_session_state(session_id)
    demographics = conversation_manager.get_demographics(session_id)

    # Store the incoming user message before calling the pipeline
    conversation_manager.add_user_message(session_id, query)

    try:
        result = _orchestrator.run(
            query=query, 
            history=history, 
            session_state=session_state, 
            demographics=demographics
        )

        # Store the assistant reply and the topic
        conversation_manager.add_assistant_message(session_id, result["answer"])
        conversation_manager.add_topic(session_id, result.get("topic", "general"))
        
        # Update demographics if newly extracted
        if "demographics" in result and result["demographics"]:
            conversation_manager.update_demographics(session_id, result["demographics"])

        # Update persona if newly extracted
        if "persona" in result and result["persona"]:
            conversation_manager.update_persona(session_id, result["persona"])

        langfuse_client.flush()

        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "session_id": session_id,
            "history_length": conversation_manager.message_count(session_id),
            "demographics": conversation_manager.get_demographics(session_id),
            "persona": conversation_manager.get_persona(session_id)
        }

    except Exception as e:
        langfuse_client.flush()
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# DELETE /chat/{session_id}  — clear conversation history
# ---------------------------------------------------------------------------

@router.delete("/chat/{session_id}")
async def clear_session(session_id: str):
    """Clears the conversation history for a given session."""
    conversation_manager.clear(session_id)
    return {"status": "cleared", "session_id": session_id}
