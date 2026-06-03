"""
app/conversation.py

In-memory sliding-window conversation history manager.

Each session keeps at most MAX_HISTORY_MESSAGES messages (user + assistant).
Oldest messages are dropped automatically when the window is full.
"""

from collections import deque
from typing import List, Dict
from app.config import settings


class ConversationManager:
    """
    Maintains per-session message history using a sliding window.

    Storage structure:
        _sessions: { session_id: deque([{role, content}, ...]) }

    The deque's maxlen enforces the sliding window automatically —
    appending beyond capacity silently drops the oldest message.
    """

    def __init__(self):
        self._sessions: Dict[str, deque] = {}
        self._demographics: Dict[str, Dict[str, str]] = {}
        self._session_states: Dict[str, dict] = {}

    def _get_or_create(self, session_id: str) -> deque:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(
                maxlen=settings.MAX_HISTORY_MESSAGES
            )
        return self._sessions[session_id]

    def get_session_state(self, session_id: str) -> dict:
        if session_id not in self._session_states:
            self._session_states[session_id] = {
                "persona": "exploring",
                "topics_covered": [],
                "state_detected": None,
                "phone_collected": False
            }
        return self._session_states[session_id]

    def add_user_message(self, session_id: str, content: str) -> None:
        self._get_or_create(session_id).append(
            {"role": "user", "content": content}
        )

    def add_assistant_message(self, session_id: str, content: str) -> None:
        self._get_or_create(session_id).append(
            {"role": "assistant", "content": content}
        )

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        return list(self._get_or_create(session_id))

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._session_states:
            del self._session_states[session_id]
        if session_id in self._demographics:
            del self._demographics[session_id]

    def message_count(self, session_id: str) -> int:
        return len(self._get_or_create(session_id))

    def update_demographics(self, session_id: str, info: Dict[str, str]) -> None:
        if session_id not in self._demographics:
            self._demographics[session_id] = {}
        
        state = self.get_session_state(session_id)
        for k, v in info.items():
            if v:
                self._demographics[session_id][k] = v
                if "phone" in k.lower():
                    state["phone_collected"] = True
                if "state" in k.lower() or k.lower() == "state_detected":
                    state["state_detected"] = v

    def get_demographics(self, session_id: str) -> Dict[str, str]:
        return self._demographics.get(session_id, {})
        
    def get_phone_collected(self, session_id: str) -> bool:
        return self.get_session_state(session_id)["phone_collected"]
        
    def add_topic(self, session_id: str, topic: str) -> None:
        state = self.get_session_state(session_id)
        if topic and topic not in state["topics_covered"]:
            state["topics_covered"].append(topic)
            
    def get_topics(self, session_id: str) -> List[str]:
        return self.get_session_state(session_id)["topics_covered"]

    def update_persona(self, session_id: str, persona: str) -> None:
        if persona in ["ready", "exploring", "comparing"]:
            self.get_session_state(session_id)["persona"] = persona

    def get_persona(self, session_id: str) -> str:
        return self.get_session_state(session_id)["persona"]

# Module-level singleton — shared across all requests in the same process
conversation_manager = ConversationManager()
