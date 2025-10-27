from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta

from utils.logger import logger

from pykrx import stock
import pandas as pd
import time
import asyncio
from pydantic import BaseModel
from typing import List

router = APIRouter(
    prefix="/api",       
    tags=["Market Data"] 
)

# --- 캐시 및 헬퍼 함수 ---
cached_data = {}
CACHE_DURATION_SECONDS = 60

def get_latest_trading_day_str():
    """가장 최근 거래일을 YYYYMMDD 문자열로 반환"""
    today = datetime.now()
    if today.weekday() >= 5: # 주말이면 금요일로
        today -= timedelta(days=today.weekday() - 4)
    while True:
        try:
            # KRX는 공휴일 데이터를 제공하지 않으므로, 거래가 있었던 날을 찾을 때까지 하루씩 빼면서 확인
            if not stock.get_market_ohlcv(today.strftime("%Y%m%d")).empty:
                 return today.strftime("%Y%m%d")
            today -= timedelta(days=1)
        except Exception:
            today -= timedelta(days=1)

# --- 통합 대시보드 API ---
@router.get("/dashboard")
async def get_dashboard_data():
    """대시보드에 필요한 모든 데이터를 한 번에 조회하여 반환"""
    global cached_data
    current_time = time.time()

    if 'dashboard' in cached_data and current_time - cached_data['dashboard']['timestamp'] < CACHE_DURATION_SECONDS:
        logger.info("캐시된 대시보드 데이터를 반환합니다.")
        return cached_data['dashboard']['data']

    try:
        logger.info("새로운 대시보드 데이터를 요청합니다.")
        latest_day = get_latest_trading_day_str()
        df_ohlcv = stock.get_market_ohlcv(latest_day, market="ALL")
        
        indices_data = {}
        today_str = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=21)).strftime('%Y%m%d')
        
        for index_name, index_code in [("kospi", "1001"), ("kosdaq", "2001")]:
            try:
                logger.info(f"{index_name} 지수 조회 시작: {start_date} ~ {today_str}")
                
                df_daily = stock.get_index_ohlcv(start_date, today_str, index_code, "d")
                
                if df_daily.empty or len(df_daily) < 2:
                    raise ValueError(f"{index_name} 일봉 데이터가 부족: {len(df_daily)}개")
                
                chart_data = [{'value': float(row['종가'])} for idx, row in df_daily.tail(7).iterrows()]
                
                latest_row = df_daily.iloc[-1]
                previous_row = df_daily.iloc[-2]
                latest_close = float(latest_row['종가'])
                previous_close = float(previous_row['종가'])
                
                latest_info = {
                    "value": round(latest_close, 2),
                    "changeValue": round(latest_close - previous_close, 2),
                    "changeRate": round((latest_close / previous_close - 1) * 100, 2)
                }
                
                logger.info(f"{index_name} 최종 데이터: 차트 개수 {len(chart_data)}")
                
            except Exception as e:
                logger.error(f"{index_name} 지수 데이터 처리 중 오류: {e}")
                try:
                    old_start = "20241001"
                    logger.info(f"{index_name} Fallback 시도: {old_start} ~ {today_str}")
                    df_fallback = stock.get_index_ohlcv(old_start, today_str, index_code, "d")
                    
                    if df_fallback.empty or len(df_fallback) < 2:
                        raise ValueError("Fallback 데이터도 부족")
                    
                    chart_data = [{'value': float(row['종가'])} for idx, row in df_fallback.tail(7).iterrows()]
                    
                    latest_row = df_fallback.iloc[-1]
                    previous_row = df_fallback.iloc[-2]
                    
                    latest_info = {
                        "value": round(float(latest_row['종가']), 2),
                        "changeValue": round(float(latest_row['종가'] - previous_row['종가']), 2),
                        "changeRate": round(float((latest_row['종가'] / previous_row['종가'] - 1) * 100), 2)
                    }
                    
                    logger.info(f"{index_name} Fallback 성공: 차트 개수 {len(chart_data)}")
                    
                except Exception as fallback_error:
                    logger.error(f"{index_name} Fallback 실패: {fallback_error}")
                    chart_data = []
                    latest_info = {"value": 0, "changeValue": 0, "changeRate": 0}
            
            indices_data[index_name] = {**latest_info, "chartData": chart_data}

        # 상승률 상위 5개
        top_gainers = df_ohlcv.sort_values(by='등락률', ascending=False).head(5)
        top_gainers_data = [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "price": row['종가'], "change_rate": round(row['등락률'], 2)} for ticker, row in top_gainers.iterrows()]

        # 하락률 상위 5개
        top_losers = df_ohlcv.sort_values(by='등락률', ascending=True).head(5)
        top_losers_data = [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "price": row['종가'], "change_rate": round(row['등락률'], 2)} for ticker, row in top_losers.iterrows()]

        # 거래량 상위 5개
        top_volume = df_ohlcv.sort_values(by='거래량', ascending=False).head(5)
        top_volume_data = [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "volume": row['거래량']} for ticker, row in top_volume.iterrows()]

        # 시가총액 상위 10개
        df_cap = stock.get_market_cap(latest_day, market="ALL")
        top_10_tickers = df_cap.sort_values(by='시가총액', ascending=False).head(10).index.tolist()
        
        top_market_cap_data = [
            {
                "code": ticker,
                "name": stock.get_market_ticker_name(ticker),
                "price": df_ohlcv.loc[ticker]['종가'],
                "change_rate": round(df_ohlcv.loc[ticker]['등락률'], 2)
            }
            for ticker in top_10_tickers if ticker in df_ohlcv.index
        ]

        dashboard_data = {
            "indices": indices_data,
            "topGainers": top_gainers_data,
            "topLosers": top_losers_data,
            "topVolume": top_volume_data,
            "topMarketCap": top_market_cap_data,
        }

        cached_data['dashboard'] = {"data": dashboard_data, "timestamp": current_time}
        return dashboard_data
        
    except Exception as e:
        logger.error(f"대시보드 데이터 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="대시보드 데이터를 가져오는 중 오류가 발생했습니다.")

async def fetch_top_gainers_data():
    """상승률 상위 5개 종목 조회 내부 함수"""
    latest_day = get_latest_trading_day_str()
    df = stock.get_market_ohlcv(latest_day, market="ALL")
    top_5 = df.sort_values(by='등락률', ascending=False).head(5)
    return [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "price": row['종가'], "change_rate": round(row['등락률'], 2)} for ticker, row in top_5.iterrows()]

async def fetch_top_losers_data():
    """하락률 상위 5개 종목 조회 내부 함수"""
    latest_day = get_latest_trading_day_str()
    df = stock.get_market_ohlcv(latest_day, market="ALL")
    top_5 = df.sort_values(by='등락률', ascending=True).head(5)
    return [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "price": row['종가'], "change_rate": round(row['등락률'], 2)} for ticker, row in top_5.iterrows()]

async def fetch_top_volume_data():
    """거래량 상위 5개 종목 조회 내부 함수"""
    latest_day = get_latest_trading_day_str()
    df = stock.get_market_ohlcv(latest_day, market="ALL")
    top_5 = df.sort_values(by='거래량', ascending=False).head(5)
    return [{"code": ticker, "name": stock.get_market_ticker_name(ticker), "volume": row['거래량']} for ticker, row in top_5.iterrows()]

async def fetch_top_market_cap_data():
    """시가총액 상위 10개 종목 조회 내부 함수 (가장 안정적인 방식)"""
    latest_day = get_latest_trading_day_str()
    
    # 1. 시가총액 보고서로 상위 10개 종목의 '코드'만 가져옵니다.
    df_cap = stock.get_market_cap(latest_day, market="ALL")
    top_10_tickers = df_cap.sort_values(by='시가총액', ascending=False).head(10).index.tolist()
    
    # 2. 전체 시장의 '가격 보고서'를 가져옵니다.
    df_ohlcv = stock.get_market_ohlcv(latest_day, market="ALL")
    
    result = []
    # 3. 상위 10개 코드에 해당하는 가격 정보만 '가격 보고서'에서 찾아와 조합합니다.
    for ticker in top_10_tickers:
        if ticker in df_ohlcv.index:
            row = df_ohlcv.loc[ticker]
            result.append({
                "code": ticker,
                "name": stock.get_market_ticker_name(ticker),
                "price": row['종가'],
                "change_rate": round(row['등락률'], 2)
            })
    return result

class TickersRequest(BaseModel):
    tickers: List[str]

@router.post("/stock-details")
async def get_stock_details(request: TickersRequest):
    """
    요청받은 종목 코드(티커) 리스트에 대한
    최신 시세 정보(종목명, 현재가, 등락률)를 반환합니다.
    """
    try:
        # 요청된 티커가 없으면 빈 리스트 반환
        if not request.tickers:
            return []
            
        latest_day = get_latest_trading_day_str()
        
        # 전체 시장의 최신 시세 정보를 한 번만 가져옵니다.
        df = stock.get_market_ohlcv(latest_day, market="ALL")
        
        # 요청받은 티커에 해당하는 데이터만 필터링합니다.
        filtered_df = df[df.index.isin(request.tickers)]
        
        result = []
        for ticker in request.tickers:
            if ticker in filtered_df.index:
                row = filtered_df.loc[ticker]
                result.append({
                    "id": ticker,
                    "name": stock.get_market_ticker_name(ticker),
                    "price": row['종가'],
                    "changePct": round(row['등락률'], 2)
                })
        return result

    except Exception as e:
        logger.error(f"개별 종목 상세 정보 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail="개별 종목 정보를 가져오는 중 오류가 발생했습니다.")

@router.get("/stock/{ticker}")
async def get_stock_detail(ticker: str):
    """
    특정 종목(ticker)의 최신 시세 정보 (OHLC, 등락률 등)를 반환합니다.
    """
    try:
        latest_day = get_latest_trading_day_str()
        
        df = stock.get_market_ohlcv(latest_day, latest_day, ticker)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="해당 종목의 데이터를 찾을 수 없습니다.")
        
        latest_data = df.iloc[0]
        
        return {
            "name": stock.get_market_ticker_name(ticker),
            "ticker": ticker,
            "price": latest_data["종가"],
            "changePct": round(latest_data["등락률"], 2),
            "ohlc": {
                "open": latest_data["시가"],
                "high": latest_data["고가"],
                "low": latest_data["저가"],
            }
        }
    except Exception as e:
        logger.error(f"종목 상세({ticker}) 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/{ticker}/chart")
async def get_stock_chart(ticker: str):
    """특정 종목의 최근 1주일간의 종가 데이터를 차트용으로 반환합니다."""
    try:
        start_date = (datetime.now() - timedelta(days=14)).strftime('%Y%m%d')
        today = datetime.now().strftime('%Y%m%d')
        
        df = stock.get_market_ohlcv(start_date, today, ticker)
        
        chart_data = df['종가'].tolist()
        
        return {"chart": chart_data}
    except Exception as e:
        logger.error(f"종목 차트({ticker}) 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=str(e))
