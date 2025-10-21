"""
주가 분석 체인 - API 데이터 조회 및 LLM 분석
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import settings
from utils.logger import logger

def query_stock_analysis(question: str, stock_data: dict):
    """
    주가 데이터를 기반으로 질문에 답변
    
    Args:
        question: 사용자 질문
        stock_data: Yahoo Finance 등에서 조회한 주가 데이터
            예: {"ticker": "005930.KS", "price": 75000, "change": +2.5, "volume": 15000000}
    
    Returns:
        답변 문자열
    """
    logger.info(f"주가 분석 질의: {question}")
    
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
        openai_api_key=settings.openai_api_key
    )
    
    # 프롬프트 템플릿
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

답변:
"""
    )
    
    # 체인 구성
    chain = prompt | llm | StrOutputParser()
    
    # 주가 데이터를 문자열로 변환
    stock_str = "\n".join([f"- {k}: {v}" for k, v in stock_data.items()])
    
    # 실행
    answer = chain.invoke({
        "question": question,
        "stock_data": stock_str
    })
    
    logger.info("주가 분석 답변 생성 완료")
    return answer
