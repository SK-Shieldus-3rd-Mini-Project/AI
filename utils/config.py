# 파일 생성: utils/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    """환경 설정"""
    
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    
    # FastAPI
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = True
    
    # ChromaDB
    chroma_db_path: str = "./embeddings/chromadb"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Embedding
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    
    class Config:
        env_file = ".env"
        case_sensitive = False

# 전역 설정 객체
settings = Settings()
