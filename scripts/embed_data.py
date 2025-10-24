# scripts/embed_data.py 수정

import sys
from pathlib import Path

# 프로젝트 루트(ai 폴더)를 sys.path에 추가 (utils 임포트 위함)
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# --- data_loader 에서 load_reports_from_deepsearch 함수 import ---
from utils.data_loader import load_reports_from_deepsearch # load_pdf 대신 사용
from utils.text_splitter import split_documents
from utils.db_client import create_vectorstore
from utils.logger import logger
# from utils.config import settings # 설정은 data_loader 내부에서 사용하므로 여기선 직접 필요 X

def main():
    """Deep Search API에서 리포트를 로드, 분할하여 ChromaDB에 임베딩"""

    # --- Deep Search API 호출 ---
    # 필요에 따라 검색어, 기간, 개수 등 파라미터 조정
    logger.info("Deep Search API를 통해 증권사 리포트 로딩 시작...")
    all_docs = load_reports_from_deepsearch(
        query="최신 증권사 리포트", # 검색어 예시
        limit=100 # 가져올 리포트 개수 예시
        
    )

    if not all_docs:
        logger.error("오류: Deep Search에서 로드된 문서가 없습니다. 임베딩을 진행할 수 없습니다.")
        return

    logger.info(f"총 {len(all_docs)}개 리포트 로드 완료. 텍스트 분할 시작...")
    # 로드된 문서들을 설정된 크기의 청크로 분할
    chunks = split_documents(all_docs)
    logger.info(f"텍스트 분할 완료 ({len(chunks)}개 청크 생성). 임베딩 및 ChromaDB 저장 시작...")

    try:
        # 분할된 청크들을 임베딩하여 ChromaDB에 저장
        # 컬렉션 이름은 'analyst_reports' (rag_chain.py와 일치시킴)
        create_vectorstore(chunks, collection_name="analyst_reports")
        logger.info("데이터 임베딩 및 ChromaDB 저장 완료!")
    except Exception as e:
        logger.error(f"ChromaDB 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    # 스크립트 실행 시 main 함수 호출
    main()