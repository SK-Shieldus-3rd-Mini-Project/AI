"""
문서 로딩 모듈
PDF, CSV, JSON 등 다양한 형식의 문서를 LangChain 형식으로 로드
"""
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from typing import List
import json
import requests
from utils.logger import logger
from utils.config import settings

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

def load_reports_from_deepsearch(
    query: str = "증권사 리포트", # 검색어 (필요시 조정)
    start_date: str = None, # 검색 시작일 (YYYY-MM-DD 형식)
    end_date: str = None, # 검색 종료일 (YYYY-MM-DD 형식)
    limit: int = 10 # 가져올 리포트 개수 제한
) -> List[Document]:
    """
    Deep Search API를 사용하여 증권사 리포트 검색 및 내용을 Document 리스트로 반환
    (주의: 실제 Deep Search API 엔드포인트 및 파라미터 확인 필요!)

    Args:
        query: 검색어
        start_date: 검색 시작일
        end_date: 검색 종료일
        limit: 최대 결과 개수

    Returns:
        Document 객체 리스트 (각 리포트가 하나의 Document)
    """
    if not settings.deepsearch_api_key:
        logger.error("Deep Search API 키가 설정되지 않았습니다.")
        return []


    api_url = "deepsearch.api.base-url=https://api-v2.deepsearch.com" 

    headers = {
        "Authorization": f"Bearer {settings.deepsearch_api_key}", # API 키 헤더
        "Content-Type": "application/json"
    }

    # 요청 파라미터 구성 (실제 API 규격에 맞게 수정)
    params = {
        "query": query,
        "document_type": "analyst_report", # 리포트 타입 지정 (가정)
        "limit": limit,
    }
    if start_date:
        params["start_date"] = start_date
    if end_date:
        params["end_date"] = end_date

    logger.info(f"Deep Search API 호출 시작: {query} (limit: {limit})")

    documents = []
    try:
        # API 호출 (GET 요청 예시, POST일 수도 있음)
        response = requests.get(api_url, headers=headers, params=params, timeout=30) # timeout 설정
        response.raise_for_status() # 오류 발생 시 예외 발생
        data = response.json()

        # --- 응답 데이터 처리 (실제 응답 구조에 맞게 수정) ---
        # 예시: 응답이 {'reports': [{'title': '...', 'content': '...', 'metadata': {...}}]} 형태라고 가정
        report_list = data.get("reports", [])
        logger.info(f"Deep Search API 응답 수신: {len(report_list)}개 리포트")

        for report in report_list:
            # 리포트 내용을 page_content 로 사용
            content = report.get("content", "")
            # 제목, 증권사, 날짜 등 메타데이터 추출
            metadata = report.get("metadata", {})
            # LangChain Document 객체 생성
            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

    except requests.exceptions.RequestException as e:
        logger.error(f"Deep Search API 호출 중 오류 발생: {e}")
    except Exception as e:
        logger.error(f"Deep Search 응답 처리 중 오류 발생: {e}")

    logger.info(f"Deep Search 리포트 로딩 완료: {len(documents)}개 Document 생성")
    return documents