import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

# In-memory session store: {session_id: [{"role": ..., "content": ...}]}
_sessions: Dict[str, List[dict]] = defaultdict(list)
_session_meta: Dict[str, dict] = defaultdict(dict)

MAX_HISTORY_TURNS = 10  # Keep last N turns per session


def add_message(session_id: str, role: str, content: str):
    """Add a message to session history."""
    _sessions[session_id].append(
        {"role": role, "content": content, "timestamp": datetime.utcnow().isoformat()}
    )
    # Trim to max history
    if len(_sessions[session_id]) > MAX_HISTORY_TURNS * 2:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY_TURNS * 2:]

    if session_id not in _session_meta:
        _session_meta[session_id] = {"created_at": datetime.utcnow().isoformat()}
    _session_meta[session_id]["last_active"] = datetime.utcnow().isoformat()


def get_history(session_id: str) -> List[dict]:
    """Return conversation history for a session (role + content only)."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in _sessions.get(session_id, [])
    ]


def get_history_as_text(session_id: str) -> str:
    """Format history as a readable string for LLM context."""
    history = _sessions.get(session_id, [])
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:  # Last 3 turns
        prefix = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{prefix}: {msg['content']}")
    return "\n".join(lines)


def clear_session(session_id: str):
    """Clear session history."""
    if session_id in _sessions:
        del _sessions[session_id]
    if session_id in _session_meta:
        del _session_meta[session_id]


def get_all_sessions() -> List[dict]:
    """List all active sessions (for admin view)."""
    sessions = []
    for sid, meta in _session_meta.items():
        sessions.append(
            {
                "session_id": sid,
                "message_count": len(_sessions.get(sid, [])),
                **meta,
            }
        )
    return sessions
