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
from utils.spring_client import spring_client

# 체인들
from chains.classifier import classify_question
from chains.rag_chain import query_rag
from chains.indicator_chain import query_economic_indicator
from chains.stock_chain import query_stock_analysis
from chains.general_chain import query_general_advice

# 라우터
from routers import market_data

# FastAPI 앱 초기화
app = FastAPI(
    title="전봉준 AI 투자 어드바이저 API",
    description="RAG 기반 투자 상담 API",
    version="1.0.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(market_data.router)

# ===== 요청/응답 모델 =====

class QueryRequest(BaseModel):
    """질문 요청 모델"""
    session_id: str
    question: str

class Source(BaseModel):
    """출처 정보 모델"""
    title: str
    securities_firm: str = "Unknown"
    date: str

class QueryResponse(BaseModel):
    """질문 응답 모델"""
    session_id: str
    question: str
    answer: str
    category: str
    sources: List[Dict]
    timestamp: str

# ===== API 엔드포인트 =====

@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "ok",
        "service": "InvestAI Core",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/ai/query", response_model=QueryResponse)
async def query_ai(request: QueryRequest):
    """
    AI 질문 처리 메인 엔드포인트
    
    흐름:
    1. 질문 분류 (+ 종목 코드 추출)
    2. 카테고리별 처리
    3. 답변 생성
    4. 응답 반환
    """
    try:
        logger.info(f"[{request.session_id}] 질문 수신: {request.question}")
        
        # ★ 1. 질문 분류 (카테고리 + 종목 코드) - async 지원
        classification = await classify_question(request.question)
        category = classification["category"]
        stock_code = classification.get("stock_code")
        
        logger.info(f"[{request.session_id}] 분류: {category}, 종목: {stock_code}")
        
        # ★ 2. 카테고리별 처리
        answer = ""
        sources = []
        
        if category == "analyst_report":
            # ★ RAG: ChromaDB 검색 + LLM 답변
            result = query_rag(request.question)
            answer = result["answer"]
            sources = result["sources"]
            
        elif category == "economic_indicator":
            # ★ 경제지표: Spring Boot DB 조회 + LLM 해석
            answer = await query_economic_indicator(request.question)
            sources = [{
                "title": "한국은행 경제통계", 
                "securities_firm": "MariaDB", 
                "date": datetime.now().strftime("%Y-%m-%d")
            }]
            
        elif category == "stock_price":
            # ★ 주가: pykrx API 조회 + LLM 분석
            if stock_code:
                answer = query_stock_analysis(request.question, stock_code)
                sources = [{
                    "title": f"실시간 주가 ({stock_code})", 
                    "securities_firm": "pykrx", 
                    "date": datetime.now().strftime("%Y-%m-%d")
                }]
            else:  # general
            # ★ 일반 상담: LLM 직접 답변
                answer = query_general_advice(request.question)
                sources = []
        
        # ★ 빈 답변 검증
            if not answer or len(answer.strip()) == 0:
                logger.error(f"[{request.session_id}] 빈 답변 생성됨. Category: {category}")
                raise HTTPException(status_code=500, detail="답변 생성 실패")
        
        # ★ 3. 응답 생성
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
        
    except HTTPException:
        raise  # HTTPException은 그대로 전달
    except Exception as e:
        logger.error(f"[{request.session_id}] 예상치 못한 오류 발생: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"AI 처리 중 오류: {str(e)}")



# ===== 서버 종료 시 정리 =====
@app.on_event("shutdown")
async def shutdown_event():
    """서버 종료 시 Spring Boot 클라이언트 정리"""
    await spring_client.close()
    logger.info("서버 종료")

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