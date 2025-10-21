"""
텍스트 분할 모듈
긴 문서를 작은 청크로 분할하여 임베딩 및 검색 효율 향상
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
from utils.config import settings
from utils.logger import logger

def split_documents(documents: List[Document]) -> List[Document]:
    """
    문서를 청크 단위로 분할
    
    Args:
        documents: Document 객체 리스트
    
    Returns:
        분할된 Document 리스트
    """
    logger.info(f"텍스트 분할 시작: {len(documents)}개 문서")
    
    # RecursiveCharacterTextSplitter: 문단 → 문장 → 단어 순으로 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,  # 각 청크의 최대 크기 (문자 수)
        chunk_overlap=settings.chunk_overlap,  # 청크 간 오버랩 (문맥 유지)
        length_function=len,  # 길이 측정 함수
        separators=["\n\n", "\n", " ", ""]  # 분할 우선순위
    )
    
    # 문서 분할 실행
    chunks = text_splitter.split_documents(documents)
    
    logger.info(f"텍스트 분할 완료: {len(chunks)}개 청크 생성")
    return chunks
