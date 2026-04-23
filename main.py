"""
RAG Customer Support Assistant — FastAPI Backend
"""
import os
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
import aiofiles

from config import settings
from models.schemas import (
    ChatRequest, ChatResponse, FeedbackRequest, FeedbackResponse,
    UploadResponse, KnowledgeBaseStats, SourceDoc, IntentType,
)
from rag.ingestion import ingest_document, load_vectorstore_from_disk, get_kb_stats
from workflow.graph import process_query
from memory.session import get_all_sessions, clear_session

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s → %(message)s",
)
logger = logging.getLogger(__name__)

# ─── Persistent Stores ────────────────────────────────────────────────────────
FEEDBACK_FILE = Path("../logs/feedback.json")
ESCALATIONS_FILE = Path("../logs/escalations.json")
FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def load_json_list(path: Path) -> list:
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return []


def save_json_list(path: Path, data: list):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


# ─── App Factory ──────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Customer Support Assistant",
    description="AI-powered customer support with RAG, LangGraph, and Groq",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount frontend
frontend_path = Path("../frontend")
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")


# ─── Startup ──────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting RAG Customer Support Assistant...")
    load_vectorstore_from_disk()
    logger.info("✅ Vectorstore loaded. Ready to serve requests.")


# ─── Auth Dependency ──────────────────────────────────────────────────────────
async def verify_admin(x_admin_key: str = Header(None)):
    if x_admin_key != settings.ADMIN_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Invalid admin key.")
    return True


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    index_path = Path("../frontend/index.html")
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "RAG Customer Support Assistant API", "version": "1.0.0"}


@app.get("/admin")
async def serve_admin():
    admin_path = Path("../frontend/admin.html")
    if admin_path.exists():
        return FileResponse(str(admin_path))
    raise HTTPException(status_code=404, detail="Admin panel not found.")


@app.post("/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    _: bool = Depends(verify_admin),
):
    """Admin endpoint: Upload and ingest a PDF or TXT document."""
    allowed_types = {".pdf", ".txt", ".md"}
    ext = Path(file.filename).suffix.lower()

    if ext not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_types}",
        )

    max_bytes = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    contents = await file.read()
    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    save_path = UPLOAD_DIR / safe_name

    async with aiofiles.open(save_path, "wb") as f:
        await f.write(contents)

    success, num_chunks = ingest_document(str(save_path), file.filename)

    if not success:
        save_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Document ingestion failed.")

    return UploadResponse(
        success=True,
        filename=file.filename,
        chunks_created=num_chunks,
        message=f"Successfully ingested '{file.filename}' into {num_chunks} chunks.",
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint: Process user query through the RAG + LangGraph pipeline."""
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = request.session_id or str(uuid.uuid4())

    try:
        result = process_query(
            session_id=session_id,
            user_message=request.message,
            user_name=request.user_name or "User",
        )
    except Exception as e:
        logger.error(f"Chat processing error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An error occurred processing your request.")

    # Log escalations
    if result["escalated"]:
        escalations = load_json_list(ESCALATIONS_FILE)
        escalations.append({
            "session_id": session_id,
            "query": request.message,
            "timestamp": datetime.utcnow().isoformat(),
            "resolved": False,
        })
        save_json_list(ESCALATIONS_FILE, escalations)

    return ChatResponse(
        session_id=session_id,
        message=result["response"],
        intent=result["intent"],
        confidence=result["confidence"],
        sources=[
            SourceDoc(**s.dict()) if hasattr(s, "dict") else SourceDoc(**s)
            for s in result["sources"]
        ],
        escalated=result["escalated"],
    )


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Collect user feedback on responses."""
    feedback_list = load_json_list(FEEDBACK_FILE)
    feedback_list.append({
        "session_id": request.session_id,
        "message_id": request.message_id,
        "rating": request.rating,
        "comment": request.comment,
        "timestamp": datetime.utcnow().isoformat(),
    })
    save_json_list(FEEDBACK_FILE, feedback_list)
    logger.info(f"Feedback received: {request.rating} for session {request.session_id}")
    return FeedbackResponse(success=True, message="Thank you for your feedback!")


# ─── Admin APIs ───────────────────────────────────────────────────────────────

@app.get("/admin/stats")
async def get_stats(_: bool = Depends(verify_admin)):
    """Admin: Get knowledge base statistics."""
    kb = get_kb_stats()
    escalations = load_json_list(ESCALATIONS_FILE)
    feedback = load_json_list(FEEDBACK_FILE)
    sessions = get_all_sessions()

    positive = sum(1 for f in feedback if f.get("rating") == 1)
    negative = sum(1 for f in feedback if f.get("rating") == -1)

    return {
        "knowledge_base": kb,
        "escalations": {
            "total": len(escalations),
            "unresolved": sum(1 for e in escalations if not e.get("resolved")),
        },
        "feedback": {
            "total": len(feedback),
            "positive": positive,
            "negative": negative,
            "satisfaction_rate": round(positive / max(len(feedback), 1) * 100, 1),
        },
        "sessions": {
            "total": len(sessions),
            "active": sessions,
        },
    }
@app.get("/admin/sessions")
async def get_sessions(_: bool = Depends(verify_admin)):
    """Admin: Get all session histories with messages."""
    from memory.session import _sessions, _session_meta
    result = []
    for sid, messages in _sessions.items():
        meta = _session_meta.get(sid, {})
        result.append({
            "session_id": sid,
            "created_at": meta.get("created_at"),
            "last_active": meta.get("last_active"),
            "messages": messages,
        })
    return result

@app.get("/admin/escalations")
async def get_escalations(_: bool = Depends(verify_admin)):
    return load_json_list(ESCALATIONS_FILE)


@app.post("/admin/escalations/{index}/resolve")
async def resolve_escalation(index: int, agent_response: str, _: bool = Depends(verify_admin)):
    escalations = load_json_list(ESCALATIONS_FILE)
    if index >= len(escalations):
        raise HTTPException(status_code=404, detail="Escalation not found.")
    escalations[index]["resolved"] = True
    escalations[index]["agent_response"] = agent_response
    escalations[index]["resolved_at"] = datetime.utcnow().isoformat()
    save_json_list(ESCALATIONS_FILE, escalations)
    return {"success": True}


@app.delete("/admin/session/{session_id}")
async def delete_session(session_id: str, _: bool = Depends(verify_admin)):
    clear_session(session_id)
    return {"success": True, "message": f"Session {session_id} cleared."}


@app.get("/health")
async def health_check():
    kb = get_kb_stats()
    return {
        "status": "healthy",
        "knowledge_base_loaded": kb["total_chunks"] > 0,
        "total_chunks": kb["total_chunks"],
        "timestamp": datetime.utcnow().isoformat(),
    }
@app.delete("/admin/document/{doc_index}")
async def delete_document(doc_index: int, _: bool = Depends(verify_admin)):
    """Admin: Delete a document from the knowledge base."""
    from rag.ingestion import delete_document_from_kb
    success = delete_document_from_kb(doc_index)
    if not success:
        raise HTTPException(status_code=404, detail="Document not found.")
    return {"success": True, "message": "Document deleted successfully."}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
