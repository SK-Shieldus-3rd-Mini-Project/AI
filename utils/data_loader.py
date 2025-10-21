"""
문서 로딩 모듈
PDF, CSV, JSON 등 다양한 형식의 문서를 LangChain 형식으로 로드
"""
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from typing import List
import json
from utils.logger import logger

def load_pdf(file_path: str) -> List[Document]:
    """
    PDF 파일을 로드하여 Document 리스트로 반환
    
    Args:
        file_path: PDF 파일 경로
    
    Returns:
        Document 객체 리스트
    """
    logger.info(f"PDF 로딩 시작: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()  # 각 페이지가 하나의 Document
    logger.info(f"PDF 로딩 완료: {len(documents)}개 페이지")
    return documents

def load_csv(file_path: str) -> List[Document]:
    """
    CSV 파일을 로드하여 Document 리스트로 반환
    
    Args:
        file_path: CSV 파일 경로
    
    Returns:
        Document 객체 리스트
    """
    logger.info(f"CSV 로딩 시작: {file_path}")
    loader = CSVLoader(file_path)
    documents = loader.load()  # 각 행이 하나의 Document
    logger.info(f"CSV 로딩 완료: {len(documents)}개 행")
    return documents

def load_json(file_path: str) -> List[Document]:
    """
    JSON 파일을 로드하여 Document 리스트로 반환
    
    Args:
        file_path: JSON 파일 경로
    
    Returns:
        Document 객체 리스트
    """
    logger.info(f"JSON 로딩 시작: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON 데이터를 Document로 변환
    documents = []
    if isinstance(data, list):
        for item in data:
            content = item.get('content', str(item))
            metadata = {k: v for k, v in item.items() if k != 'content'}
            documents.append(Document(page_content=content, metadata=metadata))
    
    logger.info(f"JSON 로딩 완료: {len(documents)}개 항목")
    return documents
