"""
문서 로딩 모듈
PDF, CSV, JSON 등 다양한 형식의 문서를 LangChain 형식으로 로드
"""
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document
from typing import List, Optional, Dict, Any
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
    query: str = '"증권사 리포트" AND ("투자 의견" OR "목표 주가")', # 리포트 관련성 높은 검색어 예시
    start_date: Optional[str] = None, # 검색 시작일 (YYYY-MM-DD)
    end_date: Optional[str] = None, # 검색 종료일 (YYYY-MM-DD)
    limit: int = 50 # 가져올 문서 개수 (API 제한 및 비용 고려)
) -> List[Document]:
    """
    Deep Search 문서 검색 API(/v1/search/documents)를 사용하여 증권사 리포트를 검색하고
    LangChain Document 객체 리스트로 반환합니다.

    **주의:** API 엔드포인트, 파라미터, 응답 구조는 Deep Search 문서를 참조하여
           정확하게 확인 후 필요시 수정해야 합니다.

    Args:
        query (str): 검색어 (Deep Search 검색 문법 사용 가능).
        start_date (Optional[str]): 검색 시작일 (YYYY-MM-DD).
        end_date (Optional[str]): 검색 종료일 (YYYY-MM-DD).
        limit (int): 가져올 최대 문서 개수.

    Returns:
        List[Document]: 각 리포트 내용과 메타데이터를 담은 Document 객체 리스트.
    """
    api_key = settings.deepsearch_api_key # 설정에서 API 키 로드
    if not api_key:
        logger.error("Deep Search API 키가 .env 파일에 설정되지 않았습니다.")
        return [] # 빈 리스트 반환

    # --- API 엔드포인트 설정 ---
    base_url = "https://api-v2.deepsearch.com" # 백엔드와 동일한 베이스 URL
    api_endpoint_path = "/v1/search/documents" # 문서 검색 API 경로
    api_url = f"{base_url}{api_endpoint_path}" # 최종 URL

    # --- API 요청 헤더 ---
    headers = {
        "Authorization": f"Bearer {api_key}", # Bearer 토큰 인증
        "Accept": "application/json"
    }
    params: Dict[str, Any] = {
        "query": query, # 검색어
        "limit": limit, # 결과 개수
    
        "filter": "document_type:리포트 AND provider_category:증권", 
    
        "fields": "id,title,abstract_ko,content_ko,provider_name,published_date" 
    }
    # 기간 파라미터 추가 (실제 파라미터 이름 확인 필요)
    if start_date:
        params["start_date"] = start_date # 예: start_date=2024-01-01
    if end_date:
        params["end_date"] = end_date   # 예: end_date=2024-10-24

    logger.info(f"Deep Search 문서 검색 API 호출 시작: URL='{api_url}', Params={params}")

    documents = [] # 결과를 담을 리스트 초기화
    try:
        # API 호출 (GET 요청 예시, POST일 수도 있음)
        response = requests.get(api_url, headers=headers, params=params, timeout=30) # timeout 설정
        response.raise_for_status() # 오류 발생 시 예외 발생
        data = response.json()

        # --- 응답 데이터 처리 (실제 응답 구조에 맞게 수정) ---
    
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