import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama3-8b-8192")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTORSTORE_PATH: str = os.getenv("VECTORSTORE_PATH", "../data/vectorstore")
    UPLOAD_DIR: str = os.getenv("UPLOAD_DIR", "../uploads")
    ADMIN_SECRET_KEY: str = os.getenv("ADMIN_SECRET_KEY", "admin123")
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", "4"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.35"))
    ALLOWED_ORIGINS: list = os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8000,http://127.0.0.1:8000"
    ).split(",")

settings = Settings()
