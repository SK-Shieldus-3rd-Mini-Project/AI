"""
FastAPI 메인 서버
모든 체인을 통합하여 Spring Boot와 연동
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
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

# pykrx API 추가
from pykrx import stock

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



def get_latest_trading_day_str():
    """가장 최근 거래일을 YYYYMMDD 문자열로 반환하는 헬퍼 함수"""
    today = datetime.now()
    # 장 마감(15:30) 이전이면 어제를 기준으로 함 (데이터 안정성)
    if today.hour < 15 or (today.hour == 15 and today.minute < 30):
        today = today - timedelta(days=1)
    
    # 주말 처리
    if today.weekday() == 5: # 토요일
        today = today - timedelta(days=1)
    elif today.weekday() == 6: # 일요일
        today = today - timedelta(days=2)
        
    while True:
        try:
            stock.get_market_ohlcv(today.strftime("%Y%m%d"))
            return today.strftime("%Y%m%d")
        except:
            today = today - timedelta(days=1)

@app.get("/api/indices")
async def get_indices():
    """
    pykrx를 사용하여 코스피/코스닥의 최신 지수 및 당일 분봉 데이터를 반환합니다.
    장이 열리지 않았을 경우, 가장 최근 거래일의 종가 데이터를 반환합니다.
    """
    try:
        logger.info("시장 지수 데이터 요청 수신")
        today = datetime.now().strftime('%Y%m%d')
        
        response = {}
        
        for index_name, index_code in [("kospi", "1001"), ("kosdaq", "2001")]:
            try:
                # 당일 분봉 데이터 조회 시도
                df = stock.get_index_ohlcv(today, today, index_code, "m")
                
                if df.empty:
                    # 데이터가 비어있으면 (장이 아직 안 열렸거나 끝난 직후) 일봉으로 재시도
                    raise ValueError("Minute data is empty, trying daily data.")

                latest_data = df.iloc[-1]
                
                # 그래프용 데이터 가공 (시간과 종가)
                # 인덱스(시간)를 'HH:MM' 형식의 문자열로 변환
                chart_data = [{'time': time.strftime('%H:%M'), 'value': row['종가']}
                              for time, row in df.iterrows()]
                
                latest_info = {
                    "value": round(latest_data['종가'], 2),
                    "changeValue": round(latest_data['종가'] - df.iloc[0]['시가'], 2), # 당일 시가 대비
                    "changeRate": round((latest_data['종가'] / df.iloc[0]['시가'] - 1) * 100, 2)
                }

            except Exception as e:
                logger.warning(f"{index_name} 분봉 데이터 조회 실패, 일봉 데이터로 대체합니다. 원인: {e}")
                # 분봉 조회 실패 시 (주말, 공휴일, 장 마감 후 등) 가장 최근 일봉 데이터 조회
                df = stock.get_index_ohlcv("20240101", today, index_code, "d").tail(1)
                latest_data = df.iloc[0]
                
                chart_data = [] # 장 마감 시에는 그래프 데이터를 보내지 않음 (또는 일봉 그래프를 보낼 수도 있음)
                
                latest_info = {
                    "value": round(latest_data['종가'], 2),
                    "changeValue": round(latest_data['변동폭'], 2),
                    "changeRate": round(latest_data['등락률'], 2)
                }

            response[index_name] = {**latest_info, "chartData": chart_data}
        
        logger.info("시장 지수 데이터 응답 완료")
        return response

    except Exception as e:
        logger.error(f"시장 지수 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="시장 지수 데이터를 가져오는 중 오류가 발생했습니다.")

@app.get("/api/top-gainers")
async def get_top_gainers():
    """상승률 상위 5개 종목 반환 (안정적인 방식으로 변경)"""
    try:
        latest_day = get_latest_trading_day_str()
        # ▼▼▼ get_market_ohlcv 함수 하나만 사용합니다. (가장 안정적) ▼▼▼
        df = stock.get_market_ohlcv(latest_day, market="ALL")
        top_5 = df.sort_values(by='등락률', ascending=False).head(5)
        
        result = [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "change_rate": round(row['등락률'], 2)} 
                  for ticker, row in top_5.iterrows()]
        return result
    except Exception as e:
        logger.error(f"상승률 상위 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/top-losers")
async def get_top_losers():
    """하락률 상위 5개 종목 반환 (안정적인 방식으로 변경)"""
    try:
        latest_day = get_latest_trading_day_str()
        df = stock.get_market_ohlcv(latest_day, market="ALL")
        top_5 = df.sort_values(by='등락률', ascending=True).head(5)
        
        result = [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "change_rate": round(row['등락률'], 2)}
                  for ticker, row in top_5.iterrows()]
        return result
    except Exception as e:
        logger.error(f"하락률 상위 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/top-volume")
async def get_top_volume():
    """거래량 상위 5개 종목 반환 (안정적인 방식으로 변경)"""
    try:
        latest_day = get_latest_trading_day_str()
        df = stock.get_market_ohlcv(latest_day, market="ALL")
        top_5 = df.sort_values(by='거래량', ascending=False).head(5)
        
        result = [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "volume": row['거래량']}
                  for ticker, row in top_5.iterrows()]
        return result
    except Exception as e:
        logger.error(f"거래량 상위 조회 중 오류: {e}")
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
