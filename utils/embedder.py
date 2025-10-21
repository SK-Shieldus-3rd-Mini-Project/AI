from langchain_openai import OpenAIEmbeddings
from utils.config import settings

def get_embeddings():
    """OpenAI 임베딩 모델 반환"""
    return OpenAIEmbeddings(
        openai_api_key=settings.openai_api_key,
        model=settings.embedding_model
    )
