"""
LangGraph-based workflow for RAG Customer Support Assistant.

Graph Nodes:
  classify_intent → route → [rag_node | greeting_node | escalate_node] → END
"""
import logging
from typing import TypedDict, List, Optional, Literal
from datetime import datetime

from groq import Groq
from langgraph.graph import StateGraph, END

from rag.retriever import retrieve_relevant_chunks, build_context_string
from memory.session import get_history_as_text, add_message
from models.schemas import IntentType, SourceDoc
from config import settings

logger = logging.getLogger(__name__)

# ─── Groq client ────────────────────────────────────────────────────────────
_groq_client: Optional[Groq] = None


def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=settings.GROQ_API_KEY)
    return _groq_client


def call_groq(messages: list, temperature: float = 0.3, max_tokens: int = 800) -> str:
    """Call Groq API with given messages."""
    client = get_groq_client()
    response = client.chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content.strip()


# ─── Graph State ─────────────────────────────────────────────────────────────
class GraphState(TypedDict):
    session_id: str
    user_message: str
    user_name: str
    intent: IntentType
    confidence: float
    context_texts: List[str]
    source_docs: List[SourceDoc]
    retrieval_score: float
    final_response: str
    escalated: bool
    history_text: str


# ─── Node: Classify Intent ────────────────────────────────────────────────────
def classify_intent_node(state: GraphState) -> GraphState:
    """Classify the user's intent using Groq LLM."""
    message = state["user_message"]

    prompt = f"""Classify this customer support message into exactly one category.

Categories:
- greeting: Hello, hi, hey, thanks, bye, how are you
- faq: Questions about products, services, policies, how-to, pricing, features
- complaint: Negative experiences, issues, problems, frustration, angry messages
- escalation: Explicitly asks for a human, agent, supervisor, or live support

Message: "{message}"

Reply with only one word — the category name (greeting/faq/complaint/escalation)."""

    try:
        result = call_groq(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=10,
        ).lower().strip().rstrip(".")

        intent_map = {
            "greeting": IntentType.GREETING,
            "faq": IntentType.FAQ,
            "complaint": IntentType.COMPLAINT,
            "escalation": IntentType.ESCALATION,
        }
        intent = intent_map.get(result, IntentType.FAQ)
    except Exception as e:
        logger.error(f"Intent classification error: {e}")
        intent = IntentType.FAQ

    logger.info(f"Intent classified as: {intent}")
    return {**state, "intent": intent}


# ─── Node: Router ─────────────────────────────────────────────────────────────
def router_node(state: GraphState) -> Literal["greeting_node", "rag_node", "escalate_node"]:
    """Route to the appropriate node based on intent."""
    intent = state["intent"]
    if intent == IntentType.GREETING:
        return "greeting_node"
    elif intent in [IntentType.COMPLAINT, IntentType.ESCALATION]:
        return "escalate_node"
    else:
        return "rag_node"


# ─── Node: Greeting ───────────────────────────────────────────────────────────
def greeting_node(state: GraphState) -> GraphState:
    """Handle simple greetings with a warm response."""
    message = state["user_message"].lower()
    name = state.get("user_name", "there")

    if any(w in message for w in ["bye", "goodbye", "see you", "later"]):
        response = f"Goodbye, {name}! 👋 Have a wonderful day. Feel free to come back if you need anything!"
    elif any(w in message for w in ["thank", "thanks", "appreciate"]):
        response = f"You're very welcome, {name}! 😊 I'm always here to help. Is there anything else I can assist you with?"
    else:
        response = (
            f"Hello, {name}! 👋 Welcome to our support center! I'm your AI assistant, "
            "ready to help you with any questions about our products or services. "
            "What can I assist you with today?"
        )

    return {
        **state,
        "final_response": response,
        "confidence": 1.0,
        "escalated": False,
        "source_docs": [],
    }


# ─── Node: RAG ───────────────────────────────────────────────────────────────
def rag_node(state: GraphState) -> GraphState:
    """Retrieve context and generate an answer using Groq LLM."""
    query = state["user_message"]
    history = state.get("history_text", "")

    # Retrieval
    context_texts, source_docs, retrieval_score = retrieve_relevant_chunks(query)
    context_str = build_context_string(context_texts)

    # Check if we have useful context
    if not context_texts or retrieval_score < settings.CONFIDENCE_THRESHOLD:
        response = (
            "I'm sorry, I couldn't find specific information about that in my knowledge base. "
            "This might be outside the scope of my current knowledge. "
            "Would you like me to connect you with a human support agent who can help further?"
        )
        return {
            **state,
            "context_texts": context_texts,
            "source_docs": source_docs,
            "retrieval_score": retrieval_score,
            "confidence": retrieval_score,
            "final_response": response,
            "escalated": retrieval_score < settings.CONFIDENCE_THRESHOLD * 0.5,
        }

    # Build prompt with history
    history_section = f"\n\nConversation History:\n{history}" if history else ""
    system_prompt = """You are a helpful, professional, and friendly customer support AI assistant. 
Your job is to answer customer questions accurately based ONLY on the provided context.

Rules:
- Be concise but thorough
- Use bullet points for multi-step answers
- If the context doesn't fully answer the question, say so honestly
- Never make up information not in the context
- Be empathetic and professional
- End with an offer to help further if appropriate"""

    user_prompt = f"""Context from knowledge base:
{context_str}
{history_section}

Customer Question: {query}

Please provide a helpful, accurate answer based on the context above."""

    try:
        answer = call_groq(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=800,
        )
        confidence = retrieval_score
    except Exception as e:
        logger.error(f"LLM generation error: {e}")
        answer = "I encountered an error generating a response. Please try again or contact our support team."
        confidence = 0.0

    return {
        **state,
        "context_texts": context_texts,
        "source_docs": source_docs,
        "retrieval_score": retrieval_score,
        "confidence": confidence,
        "final_response": answer,
        "escalated": False,
    }


# ─── Node: Escalate ───────────────────────────────────────────────────────────
def escalate_node(state: GraphState) -> GraphState:
    """Handle escalation — log and inform user."""
    intent = state["intent"]
    name = state.get("user_name", "there")

    if intent == IntentType.COMPLAINT:
        response = (
            f"I'm truly sorry to hear you're having a difficult experience, {name}. 😔 "
            "Your concern is important to us. I'm escalating this to our support team right away. "
            "A human agent will review your case and reach out shortly. "
            "Your query has been logged with priority status. "
            "\n\n📞 **You can also reach us directly:**\n"
            "- Email: support@company.com\n"
            "- Phone: 1-800-SUPPORT\n"
            "- Live Chat: Available Mon-Fri 9AM-6PM"
        )
    else:
        response = (
            f"Of course, {name}! I'm connecting you with a human support agent. 🙋 "
            "Your request has been queued and an agent will be with you shortly. "
            "\n\n📞 **Reach us directly:**\n"
            "- Email: support@company.com\n"
            "- Phone: 1-800-SUPPORT\n"
            "- Live Chat: Available Mon-Fri 9AM-6PM"
        )

    return {
        **state,
        "final_response": response,
        "confidence": 1.0,
        "escalated": True,
        "source_docs": [],
    }


# ─── Build Graph ──────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)

    graph.add_node("classify_intent", classify_intent_node)
    graph.add_node("greeting_node", greeting_node)
    graph.add_node("rag_node", rag_node)
    graph.add_node("escalate_node", escalate_node)

    graph.set_entry_point("classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        router_node,
        {
            "greeting_node": "greeting_node",
            "rag_node": "rag_node",
            "escalate_node": "escalate_node",
        },
    )

    graph.add_edge("greeting_node", END)
    graph.add_edge("rag_node", END)
    graph.add_edge("escalate_node", END)

    return graph.compile()


# Singleton compiled graph
_graph = None


def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


# ─── Main Entry Point ─────────────────────────────────────────────────────────
def process_query(session_id: str, user_message: str, user_name: str = "User") -> dict:
    """Process a user query through the full LangGraph workflow."""
    history_text = get_history_as_text(session_id)

    initial_state: GraphState = {
        "session_id": session_id,
        "user_message": user_message,
        "user_name": user_name,
        "intent": IntentType.UNKNOWN,
        "confidence": 0.0,
        "context_texts": [],
        "source_docs": [],
        "retrieval_score": 0.0,
        "final_response": "",
        "escalated": False,
        "history_text": history_text,
    }

    graph = get_graph()
    result = graph.invoke(initial_state)

    # Update session memory
    add_message(session_id, "user", user_message)
    add_message(session_id, "assistant", result["final_response"])

    return {
        "response": result["final_response"],
        "intent": result["intent"],
        "confidence": round(result["confidence"], 4),
        "sources": result.get("source_docs", []),
        "escalated": result.get("escalated", False),
    }
