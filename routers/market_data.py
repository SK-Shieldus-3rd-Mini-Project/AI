from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from utils.logger import logger
from pykrx import stock
import pandas as pd
import time

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
    if today.weekday() >= 5:
        today -= timedelta(days=today.weekday() - 4)
    
    # 최대 10일 전까지 거래일 찾기
    for _ in range(10):
        try:
            date_str = today.strftime("%Y%m%d")
            df = stock.get_market_ohlcv(date_str, market="KOSPI")
            if not df.empty:
                return date_str
            today -= timedelta(days=1)
        except Exception:
            today -= timedelta(days=1)
    
    # Fallback: 오늘 날짜 반환
    return datetime.now().strftime("%Y%m%d")

def safe_get_ohlcv(date_str, ticker=None, market="ALL"):
    """
    pykrx OHLCV 안전 조회 (컬럼명 에러 처리)
    """
    try:
        if ticker:
            df = stock.get_market_ohlcv(date_str, date_str, ticker)
        else:
            df = stock.get_market_ohlcv(date_str, market=market)
        
        # ★ 컬럼명 정규화 (pykrx 버전별 차이 대응)
        if not df.empty:
            df.columns = df.columns.str.strip()  # 공백 제거
        
        return df
    except Exception as e:
        logger.error(f"OHLCV 조회 실패 (date={date_str}, ticker={ticker}): {e}")
        return pd.DataFrame()

def safe_get_market_cap(date_str, market="ALL"):
    """시가총액 안전 조회"""
    try:
        df = stock.get_market_cap(date_str, market=market)
        if not df.empty:
            df.columns = df.columns.str.strip()
        return df
    except Exception as e:
        logger.error(f"시가총액 조회 실패: {e}")
        return pd.DataFrame()

# --- 통합 대시보드 API ---
@router.get("/dashboard")
async def get_dashboard_data():
    """대시보드에 필요한 모든 데이터를 한 번에 조회"""
    global cached_data
    current_time = time.time()

    if 'dashboard' in cached_data and current_time - cached_data['dashboard']['timestamp'] < CACHE_DURATION_SECONDS:
        logger.info("✅ 캐시된 대시보드 데이터 반환")
        return cached_data['dashboard']['data']

    try:
        logger.info("🔄 새로운 대시보드 데이터 요청")
        latest_day = get_latest_trading_day_str()
        logger.info(f"📅 최근 거래일: {latest_day}")

        # ★ 전체 시장 OHLCV 조회
        df_ohlcv = safe_get_ohlcv(latest_day, market="ALL")
        if df_ohlcv.empty:
            logger.error("❌ OHLCV 데이터가 비어있음")
            raise HTTPException(status_code=500, detail="시장 데이터 조회 실패")
        
        logger.info(f"✅ OHLCV 데이터 {len(df_ohlcv)}개 로드")

        # ★ 지수 데이터 (KOSPI, KOSDAQ)
        indices_data = {}
        today_str = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=5)).strftime('%Y%m%d')
        
        for index_name, index_code in [("kospi", "1001"), ("kosdaq", "2001")]:
            try:
                # 일봉 데이터로 전일 종가 가져오기
                df_daily = stock.get_index_ohlcv(start_date, today_str, index_code, "d")
                if len(df_daily) < 2:
                    raise ValueError("일봉 데이터 부족")
                
                previous_close = df_daily.iloc[-2]['종가']
                latest_price = df_daily.iloc[-1]['종가']
                
                # 분봉 데이터 시도
                try:
                    df_minute = stock.get_index_ohlcv(today_str, today_str, index_code, "m")
                    if not df_minute.empty:
                        chart_data = [{'time': time_idx.strftime('%H:%M'), 'value': row['종가']} 
                                     for time_idx, row in df_minute.iterrows()]
                    else:
                        chart_data = []
                except:
                    chart_data = []
                
                indices_data[index_name] = {
                    "value": round(latest_price, 2),
                    "changeValue": round(latest_price - previous_close, 2),
                    "changeRate": round((latest_price / previous_close - 1) * 100, 2),
                    "chartData": chart_data
                }
                logger.info(f"✅ {index_name} 지수 로드 성공")
            except Exception as e:
                logger.error(f"❌ {index_name} 지수 조회 실패: {e}")
                indices_data[index_name] = {
                    "value": 0.0,
                    "changeValue": 0.0,
                    "changeRate": 0.0,
                    "chartData": []
                }

        # ★ 상승률 TOP 5
        top_gainers = df_ohlcv.nlargest(5, '등락률')
        top_gainers_data = []
        for ticker, row in top_gainers.iterrows():
            try:
                name = stock.get_market_ticker_name(ticker)
                top_gainers_data.append({
                    "code": ticker,
                    "name": name,
                    "price": int(row['종가']),
                    "change_rate": round(row['등락률'], 2)
                })
            except Exception as e:
                logger.warning(f"종목명 조회 실패 ({ticker}): {e}")

        # ★ 하락률 TOP 5
        top_losers = df_ohlcv.nsmallest(5, '등락률')
        top_losers_data = []
        for ticker, row in top_losers.iterrows():
            try:
                name = stock.get_market_ticker_name(ticker)
                top_losers_data.append({
                    "code": ticker,
                    "name": name,
                    "price": int(row['종가']),
                    "change_rate": round(row['등락률'], 2)
                })
            except Exception as e:
                logger.warning(f"종목명 조회 실패 ({ticker}): {e}")

        # ★ 거래량 TOP 5
        top_volume = df_ohlcv.nlargest(5, '거래량')
        top_volume_data = []
        for ticker, row in top_volume.iterrows():
            try:
                name = stock.get_market_ticker_name(ticker)
                top_volume_data.append({
                    "code": ticker,
                    "name": name,
                    "volume": int(row['거래량'])
                })
            except Exception as e:
                logger.warning(f"종목명 조회 실패 ({ticker}): {e}")

        # ★ 시가총액 TOP 10
        df_cap = safe_get_market_cap(latest_day, market="ALL")
        top_market_cap_data = []
        
        if not df_cap.empty:
            top_10_tickers = df_cap.nlargest(10, '시가총액').index.tolist()
            
            for ticker in top_10_tickers:
                if ticker in df_ohlcv.index:
                    try:
                        row = df_ohlcv.loc[ticker]
                        name = stock.get_market_ticker_name(ticker)
                        top_market_cap_data.append({
                            "code": ticker,
                            "name": name,
                            "price": int(row['종가']),
                            "change_rate": round(row['등락률'], 2)
                        })
                    except Exception as e:
                        logger.warning(f"시가총액 TOP10 처리 실패 ({ticker}): {e}")

        # ★ 최종 데이터 조합
        dashboard_data = {
            "indices": indices_data,
            "topGainers": top_gainers_data,
            "topLosers": top_losers_data,
            "topVolume": top_volume_data,
            "topMarketCap": top_market_cap_data,
        }

        cached_data['dashboard'] = {"data": dashboard_data, "timestamp": current_time}
        logger.info("✅ 대시보드 데이터 생성 완료")
        return dashboard_data

    except Exception as e:
        logger.error(f"❌ 대시보드 데이터 조회 중 치명적 오류: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"대시보드 데이터 조회 실패: {str(e)}")

@router.get("/stock/{ticker}")
async def get_stock_detail(ticker: str):
    """특정 종목의 최신 시세 정보"""
    try:
        latest_day = get_latest_trading_day_str()
        df = safe_get_ohlcv(latest_day, ticker=ticker)
        
        if df.empty:
            raise HTTPException(status_code=404, detail="해당 종목의 데이터를 찾을 수 없습니다.")
        
        latest_data = df.iloc[0]
        
        return {
            "name": stock.get_market_ticker_name(ticker),
            "ticker": ticker,
            "price": int(latest_data["종가"]),
            "changePct": round(latest_data["등락률"], 2),
            "ohlc": {
                "open": int(latest_data["시가"]),
                "high": int(latest_data["고가"]),
                "low": int(latest_data["저가"]),
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"종목 상세 조회 실패 ({ticker}): {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stock/{ticker}/chart")
async def get_stock_chart(ticker: str):
    """특정 종목의 최근 1주일간의 종가 데이터"""
    try:
        start_date = (datetime.now() - timedelta(days=14)).strftime('%Y%m%d')
        today = datetime.now().strftime('%Y%m%d')
        
        df = safe_get_ohlcv(start_date, ticker=ticker)
        
        if df.empty:
            return {"chart": []}
        
        chart_data = df['종가'].tolist()
        return {"chart": chart_data}
    except Exception as e:
        logger.error(f"종목 차트 조회 실패 ({ticker}): {e}")
        raise HTTPException(status_code=500, detail=str(e))