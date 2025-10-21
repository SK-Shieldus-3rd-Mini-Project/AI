"""
임베딩 생성 모듈
텍스트를 벡터로 변환하여 의미 기반 검색 가능하게 함
"""
from langchain_openai import OpenAIEmbeddings
from utils.config import settings

def get_embeddings():
    """
    OpenAI 임베딩 모델 반환
    
    Returns:
        OpenAIEmbeddings 객체
    """
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,  # API 키
        model=settings.embedding_model  # text-embedding-3-small
    )
