"""
환경 설정 관리 모듈
.env 파일의 환경변수를 읽어서 애플리케이션 전체에서 사용
"""
from pydantic_settings import BaseSettings
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
class Settings(BaseSettings):
    """환경 설정 클래스"""
    
    # OpenAI API 설정
    openai_api_key: str  # OpenAI API 키 (필수)
    openai_model: str = "gpt-4-turbo-preview"  # 기본 모델
    # --- Deep Search API 설정 추가 ---
    deepsearch_api_key: str # .env 파일에서 읽어옴
    # FastAPI 서버 설정
    host: str = "0.0.0.0"  # 모든 네트워크 인터페이스에서 접근 가능
    port: int = 8001  # 서버 포트 (Spring Boot는 8000)
    debug: bool = True  # 개발 모드 활성화
    
    # ChromaDB 설정
    chroma_db_path: str = str(BASE_DIR / "embeddings/chromadb")  # 벡터 DB 저장 경로
    
    # 로깅 설정
    log_level: str = "INFO"  # 로그 레벨
    log_file: str = str(BASE_DIR / "logs/app.log")

   
    # 임베딩 설정
    embedding_model: str = "text-embedding-3-small"  # OpenAI 임베딩 모델
    chunk_size: int = 500  # 텍스트 청킹 크기 (토큰 단위)
    chunk_overlap: int = 50  # 청크 간 오버랩 (문맥 유지)
    
    class Config:
        env_file = str(BASE_DIR / ".env") # .env 파일 경로 명시
        case_sensitive = False # 환경 변수 이름 대소문자 구분 안 함

# 전역 설정 객체 생성
settings = Settings()
