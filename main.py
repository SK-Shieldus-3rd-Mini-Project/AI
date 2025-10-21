"""
FastAPI 메인 서버
모든 체인을 통합하여 Spring Boot와 연동
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import List, Dict

# 설정 및 유틸
from utils.config import settings
from utils.logger import logger

# 체인들
from chains.classifier import classify_question
from chains.rag_chain import query_rag
from chains.indicator_chain import query_economic_indicator
from chains.stock_chain import query_stock_analysis
from chains.general_chain import query_general_advice

# FastAPI 앱 초기화
app = FastAPI(
    title="전봉준 AI 투자 어드바이저 API",
    description="CHATBOT 기반 투자 상담 API",
    version="1.0.0"
)

# CORS 설정 (React, Spring Boot 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 요청/응답 모델 =====

class QueryRequest(BaseModel):
    """질문 요청 모델"""
    session_id: str  # 세션 ID (사용자 식별)
    question: str  # 사용자 질문

class Source(BaseModel):
    """출처 정보 모델"""
    title: str
    securities_firm: str
    date: str

class QueryResponse(BaseModel):
    """질문 응답 모델"""
    session_id: str
    question: str
    answer: str
    category: str  # 질문 카테고리
    sources: List[Dict]  # 출처 리스트
    timestamp: str

# ===== API 엔드포인트 =====

@app.get("/health")
async def health_check():
    """헬스 체크 - 서버 상태 확인"""
    return {
        "status": "ok",
        "service": "InvestAI Core",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/ai/query", response_model=QueryResponse)
async def query_ai(request: QueryRequest):
    """
    AI 질문 처리 메인 엔드포인트
    
    흐름:
    1. 질문 분류
    2. 카테고리별 처리
    3. 답변 생성
    4. 응답 반환
    """
    try:
        logger.info(f"[{request.session_id}] 질문 수신: {request.question}")
        
        # 1. 질문 분류
        category = classify_question(request.question)
        logger.info(f"[{request.session_id}] 분류: {category}")
        
        # 2. 카테고리별 처리
        answer = ""
        sources = []
        
        if category == "analyst_report":
            # RAG 체인 실행
            result = query_rag(request.question)
            answer = result["answer"]
            sources = result["sources"]
            
        elif category == "economic_indicator":
            # TODO: Spring Boot에서 경제지표 데이터 조회
            # 임시 더미 데이터
            indicator_data = {
                "기준금리": "3.5%",
                "M2 통화량": "3,450조원",
                "원/달러 환율": "1,320원"
            }
            answer = query_economic_indicator(request.question, indicator_data)
            sources = [{"title": "한국은행 데이터", "source": "DB", "date": "2025-10-21"}]
            
        elif category == "stock_price":
            # TODO: Yahoo Finance API 연동
            # 임시 더미 데이터
            stock_data = {
                "종목": "삼성전자",
                "현재가": "75,000원",
                "등락률": "+2.5%",
                "거래량": "15,000,000주"
            }
            answer = query_stock_analysis(request.question, stock_data)
            sources = [{"title": "실시간 주가", "source": "Yahoo Finance", "date": "2025-10-21"}]
            
        else:  # general
            # 일반 상담
            answer = query_general_advice(request.question)
            sources = []
        
        # 3. 응답 생성
        response = QueryResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            category=category,
            sources=sources,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"[{request.session_id}] 응답 생성 완료")
        return response
        
    except Exception as e:
        logger.error(f"[{request.session_id}] 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ===== 서버 실행 =====

if __name__ == "__main__":
    import uvicorn
    logger.info("서버 시작")
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
