import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Tuple

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import settings

logger = logging.getLogger(__name__)

# Global embeddings model (loaded once)
_embeddings = None
_vectorstore = None
_kb_meta = {"documents": [], "total_chunks": 0, "last_updated": None}


def get_embeddings() -> HuggingFaceEmbeddings:
    global _embeddings
    if _embeddings is None:
        logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
        _embeddings = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_vectorstore() -> FAISS | None:
    global _vectorstore
    return _vectorstore


def load_vectorstore_from_disk():
    """Load existing FAISS index from disk on startup."""
    global _vectorstore, _kb_meta
    vs_path = Path(settings.VECTORSTORE_PATH)
    meta_path = vs_path / "kb_meta.json"

    if (vs_path / "index.faiss").exists():
        try:
            embeddings = get_embeddings()
            _vectorstore = FAISS.load_local(
                str(vs_path), embeddings, allow_dangerous_deserialization=True
            )
            logger.info("Loaded existing FAISS vectorstore from disk.")
        except Exception as e:
            logger.warning(f"Could not load vectorstore: {e}")
            _vectorstore = None

    if meta_path.exists():
        with open(meta_path, "r") as f:
            _kb_meta.update(json.load(f))


def save_vectorstore_to_disk():
    """Persist FAISS index to disk."""
    global _vectorstore, _kb_meta
    if _vectorstore:
        vs_path = Path(settings.VECTORSTORE_PATH)
        vs_path.mkdir(parents=True, exist_ok=True)
        _vectorstore.save_local(str(vs_path))

        meta_path = vs_path / "kb_meta.json"
        meta_copy = dict(_kb_meta)
        meta_copy["last_updated"] = (
            meta_copy["last_updated"].isoformat()
            if isinstance(meta_copy["last_updated"], datetime)
            else meta_copy["last_updated"]
        )
        with open(meta_path, "w") as f:
            json.dump(meta_copy, f, indent=2)
        logger.info("Saved FAISS vectorstore to disk.")


def extract_text_from_pdf(file_path: str) -> str:
    """Extract full text from a PDF file."""
    reader = PdfReader(file_path)
    text_parts = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            text_parts.append(text)
    return "\n".join(text_parts)


def extract_text_from_txt(file_path: str) -> str:
    """Extract text from a plain text file."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunk_text(text: str, filename: str) -> List[dict]:
    """Split text into overlapping chunks with metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    docs = []
    for i, chunk in enumerate(chunks):
        docs.append(
            {
                "content": chunk,
                "metadata": {
                    "filename": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                },
            }
        )
    return docs


def ingest_document(file_path: str, filename: str) -> Tuple[bool, int]:
    """
    Full ingestion pipeline:
    1. Extract text
    2. Chunk
    3. Embed
    4. Store in FAISS
    Returns (success, num_chunks)
    """
    global _vectorstore, _kb_meta

    try:
        ext = Path(filename).suffix.lower()
        if ext == ".pdf":
            text = extract_text_from_pdf(file_path)
        elif ext in [".txt", ".md"]:
            text = extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        if not text.strip():
            raise ValueError("No text could be extracted from the document.")

        chunks = chunk_text(text, filename)
        if not chunks:
            raise ValueError("No chunks created from document.")

        embeddings = get_embeddings()
        texts = [c["content"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]

        new_vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

        if _vectorstore is None:
            _vectorstore = new_vs
        else:
            _vectorstore.merge_from(new_vs)

        # Update metadata
        _kb_meta["documents"].append(
            {
                "filename": filename,
                "chunks": len(chunks),
                "ingested_at": datetime.utcnow().isoformat(),
            }
        )
        _kb_meta["total_chunks"] = sum(
            d["chunks"] for d in _kb_meta["documents"]
        )
        _kb_meta["last_updated"] = datetime.utcnow()

        save_vectorstore_to_disk()
        logger.info(f"Ingested '{filename}' → {len(chunks)} chunks.")
        return True, len(chunks)

    except Exception as e:
        logger.error(f"Ingestion failed for '{filename}': {e}")
        return False, 0


def get_kb_stats() -> dict:
    return {
        "total_documents": len(_kb_meta["documents"]),
        "total_chunks": _kb_meta["total_chunks"],
        "last_updated": _kb_meta["last_updated"],
        "documents": _kb_meta["documents"],
    }
def delete_document_from_kb(doc_index: int) -> bool:
    """Delete a document by index and rebuild the vectorstore."""
    global _vectorstore, _kb_meta

    try:
        if doc_index < 0 or doc_index >= len(_kb_meta["documents"]):
            return False

        deleted_filename = _kb_meta["documents"][doc_index]["filename"]
        logger.info(f"Deleting document: {deleted_filename}")

        # Remove from metadata
        _kb_meta["documents"].pop(doc_index)
        _kb_meta["total_chunks"] = sum(d["chunks"] for d in _kb_meta["documents"])
        _kb_meta["last_updated"] = datetime.utcnow()

        # Rebuild vectorstore from remaining documents
        _vectorstore = None
        remaining_docs = list(_kb_meta["documents"])  # snapshot

        if not remaining_docs:
            # No docs left — clear the vectorstore files
            vs_path = Path(settings.VECTORSTORE_PATH)
            for f in vs_path.glob("*"):
                f.unlink()
            logger.info("Knowledge base is now empty.")
            return True

        # Re-ingest all remaining docs from the uploads folder
        upload_dir = Path(settings.UPLOAD_DIR)
        embeddings = get_embeddings()

        for doc in remaining_docs:
            fname = doc["filename"]
            # Find the file in uploads (it was saved with a uuid prefix)
            matches = list(upload_dir.glob(f"*_{fname}"))
            if not matches:
                logger.warning(f"Could not find upload file for '{fname}', skipping.")
                continue
            file_path = str(matches[-1])  # use most recent if duplicates
            ext = Path(fname).suffix.lower()
            if ext == ".pdf":
                text = extract_text_from_pdf(file_path)
            else:
                text = extract_text_from_txt(file_path)

            if not text.strip():
                continue

            chunks = chunk_text(text, fname)
            texts = [c["content"] for c in chunks]
            metadatas = [c["metadata"] for c in chunks]
            new_vs = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

            if _vectorstore is None:
                _vectorstore = new_vs
            else:
                _vectorstore.merge_from(new_vs)

        save_vectorstore_to_disk()
        logger.info(f"Vectorstore rebuilt after deleting '{deleted_filename}'.")
        return True

    except Exception as e:
        logger.error(f"Delete failed: {e}")
        return False