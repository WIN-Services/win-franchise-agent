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

    def _get_or_create(self, session_id: str) -> deque:
        if session_id not in self._sessions:
            self._sessions[session_id] = deque(
                maxlen=settings.MAX_HISTORY_MESSAGES
            )
        return self._sessions[session_id]

    def add_user_message(self, session_id: str, content: str) -> None:
        self._get_or_create(session_id).append(
            {"role": "user", "content": content}
        )

    def add_assistant_message(self, session_id: str, content: str) -> None:
        self._get_or_create(session_id).append(
            {"role": "assistant", "content": content}
        )

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Returns messages oldest-first, ready to prepend to the prompt."""
        return list(self._get_or_create(session_id))

    def clear(self, session_id: str) -> None:
        if session_id in self._sessions:
            del self._sessions[session_id]

    def message_count(self, session_id: str) -> int:
        return len(self._get_or_create(session_id))

    def update_demographics(self, session_id: str, info: Dict[str, str]) -> None:
        """Merges new demographic info into the session's demographics dictionary."""
        if session_id not in self._demographics:
            self._demographics[session_id] = {}
        
        # Only update keys that actually have values
        for k, v in info.items():
            if v:
                self._demographics[session_id][k] = v

    def get_demographics(self, session_id: str) -> Dict[str, str]:
        """Returns the current known demographics for the session."""
        return self._demographics.get(session_id, {})


# Module-level singleton — shared across all requests in the same process
conversation_manager = ConversationManager()
