"""
질문 분류 체인
사용자 질문을 4가지 카테고리로 자동 분류 + 종목 코드 추출
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import settings
from utils.logger import logger
import re

def classify_question(question: str) -> dict:
    """
    사용자 질문을 카테고리로 분류하고 필요 시 종목 코드 추출
    
    Args:
        question: 사용자 질문
    
    Returns:
        {"category": str, "stock_code": str (optional)}
    """
    logger.info(f"질문 분류 시작: {question}")
    
    # LLM 초기화 (temperature=0으로 일관된 분류)
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,
        openai_api_key=settings.openai_api_key
    )
    
    # ★ 분류 프롬프트: 카테고리 + 종목명 추출
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
당신은 투자 질문을 분류하는 전문가입니다.
사용자 질문을 아래 4가지 카테고리 중 **정확히 하나**로 분류하세요.

카테고리:
1. economic_indicator - 기준금리, M2, 환율, GDP 등 경제지표 관련
2. stock_price - 특정 기업의 주가, 시가총액, 거래량, 재무제표 관련
3. analyst_report - 증권사 리포트, 애널리스트 의견, 목표주가 관련
4. general - 일반적인 투자 전략, 조언, 포트폴리오 관련

질문: {question}

답변 형식: 
category: 카테고리명
stock: 종목명 (stock_price인 경우만, 없으면 none)

예시:
질문: "삼성전자 주가가 얼마야?"
답변: 
category: stock_price
stock: 삼성전자

질문: "기준금리가 주식에 미치는 영향은?"
답변:
category: economic_indicator
stock: none

답변:
"""
    )
    
    # 체인 구성
    chain = prompt | llm | StrOutputParser()
    
    # 실행
    result = chain.invoke({"question": question}).strip()
    
    # ★ 결과 파싱
    category_match = re.search(r'category:\s*(\w+)', result)
    stock_match = re.search(r'stock:\s*(.+)', result)
    
    category = category_match.group(1) if category_match else "general"
    stock_name = stock_match.group(1).strip() if stock_match else "none"
    
    # ★ 종목명 → 종목 코드 변환 (간단한 매핑, 실제로는 DB 조회)
    stock_code = None
    if stock_name != "none":
        stock_code = get_stock_code(stock_name)
    
    result_dict = {
        "category": category,
        "stock_code": stock_code
    }
    
    logger.info(f"분류 결과: {result_dict}")
    return result_dict

def get_stock_code(stock_name: str) -> str:
    """
    종목명 → 종목 코드 변환 (간단한 예시)
    실제로는 Spring Boot의 Stock 테이블 조회 권장
    
    Args:
        stock_name: 종목명
    
    Returns:
        종목 코드 (6자리)
    """
    # ★ 주요 종목 매핑 (예시)
    stock_map = {
        "삼성전자": "005930",
        "네이버": "035420",
        "현대차": "005380",
        "SK하이닉스": "000660",
        "카카오": "035720"
    }
    
    return stock_map.get(stock_name, None)