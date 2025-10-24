"""
FastAPI 메인 서버
모든 체인을 통합하여 Spring Boot와 연동
"""
from dotenv import load_dotenv
load_dotenv()

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
from chains.sentiment_analyzer import SentimentAnalyzer
from chains.user_stocks_chain import UserStocksChain
# pykrx API 추가
from pykrx import stock
import pandas as pd

# 인스턴스 생성
sentiment_analyzer = SentimentAnalyzer()
user_stocks_chain = UserStocksChain()

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

class UserRequest(BaseModel):
    """사용자 정보 요청 모델"""
    user_id: str

class QueryRequest(BaseModel):
    """질문 요청 모델"""
    session_id: str  # 세션 ID (사용자 식별)
    question: str    # 사용자 질문

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
    positive_opinion: str
    negative_opinion: str
    category: str  # 질문 카테고리
    sources: List[Dict]  # 출처 리스트
    timestamp: str

class StocksResponse(BaseModel):
    """보유 종목 응답 모델"""
    success: bool
    user_id: str
    data: Dict

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

@app.post("/api/ai/my-stocks", response_model=StocksResponse)
async def get_my_stocks(request: UserRequest):
    """
    사용자 보유 종목 조회 API
    - DB에서 USER_PORTFOLIO 기반 보유 종목 조회
    - yfinance로 실시간 주가 정보 추가
    - LLM으로 자연어 요약 생성
    """
    try:
        logger.info(f"[{request.user_id}] 보유 종목 조회 요청")
        
        # user_stocks_chain에서 데이터 조회
        result = user_stocks_chain.get_user_stocks(request.user_id)
        
        logger.info(f"[{request.user_id}] 보유 종목 조회 완료")
        
        return {
            "success": True,
            "user_id": request.user_id,
            "data": {
                "stocks": result["stocks"],
                "summary": result["summary"]
            }
        }
    except Exception as e:
        logger.error(f"[{request.user_id}] 보유 종목 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/query", response_model=QueryResponse)
async def query_ai(request: QueryRequest):
    """
    AI 질문 처리 메인 엔드포인트
    1. 질문 분류
    2. 카테고리별 처리(답변, 출처)
    3. 긍정/부정 의견 추출
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
            result = query_rag(request.question)
            answer = result["answer"]
            sources = result["sources"]

        elif category == "economic_indicator":
            indicator_data = {
                "기준금리": "3.5%",
                "M2 통화량": "3,450조원",
                "원/달러 환율": "1,320원"
            }
            answer = query_economic_indicator(request.question, indicator_data)
            sources = [{"title": "한국은행 데이터", "source": "DB", "date": "2025-10-21"}]

        elif category == "stock_price":
            stock_data = {
                "종목": "삼성전자",
                "현재가": "75,000원",
                "등락률": "+2.5%",
                "거래량": "15,000,000주"
            }
            answer = query_stock_analysis(request.question, stock_data)
            sources = [{"title": "실시간 주가", "source": "Yahoo Finance", "date": "2025-10-21"}]

        else:  # general
            answer = query_general_advice(request.question)
            sources = []

        # 3. 긍정/부정 의견 추출
        context = answer
        sentiment_result = sentiment_analyzer.analyze(
            question=request.question,
            context=context
        )

        # 4. 응답 반환
        response = QueryResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            positive_opinion=sentiment_result['positive_opinion'],
            negative_opinion=sentiment_result['negative_opinion'],
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
    if today.hour < 15 or (today.hour == 15 and today.minute < 30):
        today = today - timedelta(days=1)
    
    if today.weekday() == 5: 
        today = today - timedelta(days=1)
    elif today.weekday() == 6: 
        today = today - timedelta(days=2)
        
    while True:
        try:
            stock.get_market_ohlcv(today.strftime("%Y%m%d"))
            return today.strftime("%Y%m%d")
        except:
            today = today - timedelta(days=1)
    
    
@app.get("/ai/marketdata/indices")
async def get_indices():
    try:
        logger.info("시장 지수 데이터 요청 수신")
        today_str = datetime.now().strftime('%Y%m%d')
        # 최근 5일치 데이터를 여유롭게 가져옴
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')

        response = {}

        for index_name, index_code in [("kospi", "1001"), ("kosdaq", "2001")]:
            try:
                df_daily = stock.get_index_ohlcv(start_date, today_str, index_code, "d")
                previous_close = df_daily.iloc[-2]['종가']
                df_minute = stock.get_index_ohlcv(today_str, today_str, index_code, "m")

                if df_minute.empty:
                    raise ValueError("Minute data is empty, using daily data as fallback.")

                latest_price = df_minute.iloc[-1]['종가']

                chart_data = [{'time': time.strftime('%H:%M'), 'value': row['종가']}
                              for time, row in df_minute.iterrows()]

                latest_info = {
                    "value": round(latest_price, 2),
                    "changeValue": round(latest_price - previous_close, 2),
                    "changeRate": round((latest_price / previous_close - 1) * 100, 2)
                }

            except Exception as e:
                logger.warning(f"{index_name} 분봉 데이터 조회 실패, 최신 일봉으로 대체합니다. 원인: {e}")
                df_daily_fallback = stock.get_index_ohlcv(start_date, today_str, index_code, "d").tail(1)
                latest_daily_data = df_daily_fallback.iloc[0]

                chart_data = []

                latest_info = {
                    "value": round(latest_daily_data['종가'], 2),
                    "changeValue": round(latest_daily_data['변동폭'], 2),
                    "changeRate": round(latest_daily_data['등락률'], 2)
                }

            response[index_name] = {**latest_info, "chartData": chart_data}

        logger.info("시장 지수 데이터 응답 완료")
        return response

    except Exception as e:
        logger.error(f"시장 지수 조회 중 오류 발생: {str(e)}")
        raise HTTPException(status_code=500, detail="시장 지수 데이터를 가져오는 중 오류가 발생했습니다.")

@app.get("/ai/marketdata/top-gainers")
async def get_top_gainers():
    """상승률 상위 5개 종목 반환 (오늘 실시간 데이터)"""
    try:
        today_str = datetime.now().strftime("%Y%m%d")
        df = stock.get_market_ohlcv(today_str, market="ALL")
        top_5 = df.sort_values(by='등락률', ascending=False).head(5)

        result = []
        for ticker, row in top_5.iterrows():
            result.append({
                "code": ticker,
                "name": stock.get_market_ticker_name(ticker),
                "price": row['종가'],
                "change_rate": round(row['등락률'], 2)
            })
        return result
    except Exception as e:
        logger.error(f"상승률 상위 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/marketdata/top-losers")
async def get_top_losers():
    """하락률 상위 5개 종목 반환 (오늘 실시간 데이터)"""
    try:
        today_str = datetime.now().strftime("%Y%m%d")
        df = stock.get_market_ohlcv(today_str, market="ALL")
        top_5 = df.sort_values(by='등락률', ascending=True).head(5)

        result = []
        for ticker, row in top_5.iterrows():
            result.append({
                "code": ticker,
                "name": stock.get_market_ticker_name(ticker),
                "price": row['종가'],
                "change_rate": round(row['등락률'], 2)
            })
        return result
    except Exception as e:
        logger.error(f"하락률 상위 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/marketdata/top-volume")
async def get_top_volume():
    """거래량 상위 5개 종목 반환 (오늘 실시간 데이터)"""
    try:
        today_str = datetime.now().strftime("%Y%m%d")

        df = stock.get_market_ohlcv(today_str, market="ALL")
        top_5 = df.sort_values(by='거래량', ascending=False).head(5)

        result = []
        for ticker, row in top_5.iterrows():
            result.append({
                "code": ticker,
                "name": stock.get_market_ticker_name(ticker),
                "volume": row['거래량']
            })
        return result
    except Exception as e:
        logger.error(f"거래량 상위 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/marketdata/top-market-cap")
async def get_top_market_cap():
    """시가총액 상위 10개 종목의 시세 정보를 반환"""
    try:
        latest_day = get_latest_trading_day_str()

        df_cap = stock.get_market_cap(latest_day, market="ALL")
        top_10_tickers = df_cap.sort_values(by='시가총액', ascending=False).head(10).index.tolist()

        result = []
        for ticker in top_10_tickers:
            df_ohlcv = stock.get_market_ohlcv(latest_day, latest_day, ticker)

            if not df_ohlcv.empty:
                row = df_ohlcv.iloc[0]
                result.append({
                    "code": ticker,
                    "name": stock.get_market_ticker_name(ticker),
                    "price": row['종가'],
                    "change_rate": round(row['등락률'], 2)
                })
        return result
    except Exception as e:
        logger.error(f"시가총액 상위 조회 중 오류: {e}")
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
