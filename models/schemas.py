from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum


class IntentType(str, Enum):
    FAQ = "faq"
    GREETING = "greeting"
    COMPLAINT = "complaint"
    ESCALATION = "escalation"
    UNKNOWN = "unknown"


class ChatRequest(BaseModel):
    session_id: str
    message: str
    user_name: Optional[str] = "User"


class SourceDoc(BaseModel):
    filename: str
    chunk_index: int
    score: float
    preview: str


class ChatResponse(BaseModel):
    session_id: str
    message: str
    intent: IntentType
    confidence: float
    sources: List[SourceDoc] = []
    escalated: bool = False
    timestamp: datetime = None

    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = datetime.utcnow()
        super().__init__(**data)


class FeedbackRequest(BaseModel):
    session_id: str
    message_id: str
    rating: int  # 1 = thumbs up, -1 = thumbs down
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    success: bool
    message: str


class UploadResponse(BaseModel):
    success: bool
    filename: str
    chunks_created: int
    message: str


class EscalatedQuery(BaseModel):
    session_id: str
    query: str
    timestamp: datetime
    resolved: bool = False
    agent_response: Optional[str] = None


class KnowledgeBaseStats(BaseModel):
    total_documents: int
    total_chunks: int
    last_updated: Optional[datetime] = None
