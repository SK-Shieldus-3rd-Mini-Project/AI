"""
ChromaDB 클라이언트 모듈
벡터 DB 생성, 저장, 검색 기능 제공
"""
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from typing import List
from utils.embedder import get_embeddings
from utils.config import settings
from utils.logger import logger

def get_vectorstore(collection_name: str = "analyst_reports"):
    """
    기존 ChromaDB 벡터스토어 로드
    
    Args:
        collection_name: 컬렉션 이름
    
    Returns:
        Chroma 벡터스토어 객체
    """
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=settings.chroma_db_path,  # 저장 경로
        embedding_function=embeddings,  # 임베딩 함수
        collection_name=collection_name  # 컬렉션 이름
    )
    logger.info(f"벡터스토어 로드 완료: {collection_name}")
    return vectorstore

def create_vectorstore(documents: List[Document], collection_name: str = "analyst_reports"):
    """
    새로운 ChromaDB 벡터스토어 생성
    
    Args:
        documents: Document 리스트
        collection_name: 컬렉션 이름
    
    Returns:
        Chroma 벡터스토어 객체
    """
    logger.info(f"벡터스토어 생성 시작: {len(documents)}개 문서")
    
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=documents,  # 임베딩할 문서들
        embedding=embeddings,  # 임베딩 함수
        persist_directory=settings.chroma_db_path,  # 저장 경로
        collection_name=collection_name  # 컬렉션 이름
    )
    
    logger.info(f"벡터스토어 생성 완료: {collection_name}")
    return vectorstore
