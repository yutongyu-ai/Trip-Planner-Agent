"""In-memory chat-session store for the conversational itinerary-edit agent.

Portfolio-scope simplification: sessions live only in this process's memory
(a plain dict keyed by session_id) and are lost on restart or if the app runs
behind more than one worker process -- a real deployment would back this with
Redis/Postgres (LangGraph ships checkpointer backends for both). Called out
explicitly here rather than silently assumed to be production-grade.

A session bundles the trip's `AgentState` (mutated in place by the chat
tools in app.agent.tools as the traveller edits their itinerary) together
with the LangGraph checkpointer that holds that session's own conversation
history -- both must survive across separate /chat requests for the same
session_id, which is why they're stored together rather than the agent graph
being rebuilt from scratch with fresh memory each call.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from app.graph.state import AgentState


@dataclass
class ChatSession:
    state: AgentState
    checkpointer: BaseCheckpointSaver = field(default_factory=MemorySaver)


_SESSIONS: dict[str, ChatSession] = {}


def create_session(state: AgentState, session_id: str | None = None) -> str:
    session_id = session_id or str(uuid.uuid4())
    _SESSIONS[session_id] = ChatSession(state=state)
    return session_id


def get_session(session_id: str) -> ChatSession | None:
    return _SESSIONS.get(session_id)
