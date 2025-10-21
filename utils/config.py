from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """환경 설정 관리"""
    
    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4-turbo-preview"
    
    # Server
    host: str = "0.0.0.0"
    port: int = 8001
    debug: bool = True
    
    # ChromaDB
    chroma_db_path: str = "./embeddings/chromadb"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "./logs/app.log"
    
    # Embeddings
    embedding_model: str = "text-embedding-3-small"
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    class Config:
        env_file = ".env"
        case_sensitive = False

settings = Settings()
