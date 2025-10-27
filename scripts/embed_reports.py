"""
증권사 리포트 PDF를 임베딩하여 ChromaDB에 저장하는 스크립트
실행 방법: python scripts/embed_reports.py
"""
import sys
sys.path.append('..')  # 상위 폴더 모듈 import 가능

from utils.data_loader import load_pdf
from utils.text_splitter import split_documents
from utils.db_client import create_vectorstore
from utils.logger import logger
import glob
import os

def embed_all_reports():
    """data/reports/ 폴더의 모든 PDF를 임베딩"""
    
    # ★ 1. PDF 파일 목록 가져오기
    pdf_files = glob.glob("../data/reports/*.pdf")
    
    if not pdf_files:
        logger.warning("data/reports/ 폴더에 PDF 파일이 없습니다.")
        return
    
    logger.info(f"총 {len(pdf_files)}개의 PDF 파일 발견")
    
    all_documents = []
    
    # ★ 2. 각 PDF 로드 및 메타데이터 추가
    for pdf_file in pdf_files:
        logger.info(f"PDF 로딩: {pdf_file}")
        
        # PDF 로드
        documents = load_pdf(pdf_file)
        
        # ★ 메타데이터 추가 (파일명에서 추출)
        # 예: "NH투자증권_삼성전자_20251015.pdf"
        filename = os.path.basename(pdf_file)
        parts = filename.replace('.pdf', '').split('_')
        
        for doc in documents:
            doc.metadata = {
                "title": filename,
                "securities_firm": parts[0] if len(parts) > 0 else "Unknown",
                "company": parts[1] if len(parts) > 1 else "Unknown",
                "date": parts[2] if len(parts) > 2 else "Unknown",
                "source": pdf_file
            }
        
        all_documents.extend(documents)
    
    logger.info(f"총 {len(all_documents)}개 페이지 로드 완료")
    
    # ★ 3. 텍스트 청킹
    chunks = split_documents(all_documents)
    
    # ★ 4. 임베딩 및 ChromaDB 저장
    vectorstore = create_vectorstore(
        documents=chunks,
        collection_name="analyst_reports"
    )
    
    logger.info(f"임베딩 완료! ChromaDB에 {len(chunks)}개 청크 저장됨")
    logger.info(f"저장 경로: {vectorstore._persist_directory}")

if __name__ == "__main__":
    logger.info("증권사 리포트 임베딩 시작")
    embed_all_reports()
    logger.info("임베딩 완료!")