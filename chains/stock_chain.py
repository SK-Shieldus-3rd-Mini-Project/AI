"""
주가 분석 체인 - pykrx API 데이터 조회 및 LLM 분석
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import settings
from utils.logger import logger
from pykrx import stock
from datetime import datetime, timedelta

def get_stock_data_from_pykrx(stock_code: str) -> dict:
    """
    pykrx에서 주가 데이터 조회
    
    Args:
        stock_code: 종목 코드 (예: "005930" - 삼성전자)
    
    Returns:
        주가 데이터 딕셔너리
    """
    try:
        # ★ 최근 거래일 계산 (오늘이 주말이면 금요일로)
        today = datetime.now()
        if today.weekday() >= 5:  # 토요일(5) 또는 일요일(6)
            today -= timedelta(days=today.weekday() - 4)
        
        today_str = today.strftime("%Y%m%d")
        
        # ★ pykrx로 주가 데이터 조회
        df = stock.get_market_ohlcv(today_str, today_str, stock_code)
        
        if df.empty:
            logger.warning(f"주가 데이터 없음: {stock_code}")
            return {}
        
        latest = df.iloc[0]
        stock_name = stock.get_market_ticker_name(stock_code)
        
        # ★ 주가 데이터 구조화
        return {
            "ticker": stock_code,
            "name": stock_name,
            "price": int(latest["종가"]),
            "change_pct": round(latest["등락률"], 2),
            "volume": int(latest["거래량"]),
            "open": int(latest["시가"]),
            "high": int(latest["고가"]),
            "low": int(latest["저가"]),
        }
    except Exception as e:
        logger.error(f"pykrx 주가 조회 실패: {e}")
        return {}

def query_stock_analysis(question: str, stock_code: str):
    """
    주가 데이터를 기반으로 질문에 답변
    
    Args:
        question: 사용자 질문
        stock_code: 종목 코드
    
    Returns:
        답변 문자열
    """
    logger.info(f"주가 분석 질의: {question}, 종목: {stock_code}")
    
    # ★ pykrx에서 실시간 주가 데이터 조회
    stock_data = get_stock_data_from_pykrx(stock_code)
    
    if not stock_data:
        return "죄송합니다. 해당 종목의 주가 데이터를 조회할 수 없습니다."
    
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
        openai_api_key=settings.openai_api_key
    )
    
    # ★ 프롬프트: 주가 데이터를 컨텍스트로 제공
    prompt = PromptTemplate(
        input_variables=["question", "stock_data"],
        template="""
당신은 주식 애널리스트입니다.
아래 주가 데이터를 기반으로 질문에 답변하세요.

주가 데이터:
{stock_data}

질문: {question}

답변 지침:
1. 현재 주가와 변동률을 명확히 설명하세요
2. 거래량과 시가총액을 고려한 시장 동향을 분석하세요
3. 투자 시 주의사항을 언급하세요
4. 구체적인 매수/매도 추천은 하지 마세요

답변:
"""
    )
    
    # 체인 구성
    chain = prompt | llm | StrOutputParser()
    
    # ★ 주가 데이터를 문자열로 변환
    stock_str = "\n".join([f"- {k}: {v}" for k, v in stock_data.items()])
    
    # 실행
    answer = chain.invoke({
        "question": question,
        "stock_data": stock_str
    })
    
    logger.info("주가 분석 답변 생성 완료")
    return answer