import logging
from typing import List, Tuple, Optional

from rag.ingestion import get_vectorstore
from models.schemas import SourceDoc
from config import settings

logger = logging.getLogger(__name__)


def retrieve_relevant_chunks(
    query: str, top_k: int = None
) -> Tuple[List[str], List[SourceDoc], float]:
    """
    Retrieve top-k most relevant chunks for a query.
    Returns: (context_texts, source_docs, max_similarity_score)
    """
    vs = get_vectorstore()
    if vs is None:
        logger.warning("Vectorstore is empty — no documents ingested yet.")
        return [], [], 0.0

    k = top_k or settings.TOP_K_RESULTS
    try:
        results = vs.similarity_search_with_score(query, k=k)
    except Exception as e:
        logger.error(f"Retrieval error: {e}")
        return [], [], 0.0

    if not results:
        return [], [], 0.0

    context_texts = []
    source_docs = []
    scores = []

    for doc, score in results:
        # FAISS returns L2 distance; convert to similarity (lower is better)
        # Normalize: similarity = 1 / (1 + score)  [0-1 range]
        similarity = float(1.0 / (1.0 + score))
        scores.append(similarity)

        context_texts.append(doc.page_content)
        source_docs.append(
            SourceDoc(
                filename=doc.metadata.get("filename", "Unknown"),
                chunk_index=doc.metadata.get("chunk_index", 0),
                score=round(similarity, 4),
                preview=doc.page_content[:150] + "..."
                if len(doc.page_content) > 150
                else doc.page_content,
            )
        )

    max_score = max(scores) if scores else 0.0
    logger.info(
        f"Retrieved {len(results)} chunks for query. Max similarity: {max_score:.3f}"
    )
    return context_texts, source_docs, max_score


def build_context_string(context_texts: List[str]) -> str:
    """Format retrieved chunks into a single context block."""
    if not context_texts:
        return ""
    parts = []
    for i, text in enumerate(context_texts, 1):
        parts.append(f"[Source {i}]\n{text}")
    return "\n\n---\n\n".join(parts)
