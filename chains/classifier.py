"""
질문 분류 체인
사용자 질문을 4가지 카테고리로 자동 분류
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import settings
from utils.logger import logger

def classify_question(question: str) -> str:
    """
    사용자 질문을 카테고리로 분류
    
    Args:
        question: 사용자 질문
    
    Returns:
        카테고리 문자열 (economic_indicator, stock_price, analyst_report, general)
    """
    logger.info(f"질문 분류 시작: {question}")
    
    # LLM 초기화 (temperature=0으로 일관된 분류)
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.0,  # 결정론적 출력
        openai_api_key=settings.openai_api_key
    )
    
    # 분류 프롬프트 템플릿
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
당신은 투자 질문을 분류하는 전문가입니다.
사용자 질문을 아래 4가지 카테고리 중 **정확히 하나**로 분류하세요.

카테고리:
1. economic_indicator - 기준금리, M2, 환율, GDP 등 경제지표 관련
2. stock_price - 특정 기업의 주가, 시가총액, 재무제표 관련
3. analyst_report - 증권사 리포트, 애널리스트 의견, 목표주가 관련
4. general - 일반적인 투자 전략, 조언, 포트폴리오 관련

질문: {question}

답변 형식: 카테고리 이름만 정확히 출력하세요 (예: economic_indicator)
카테고리:
"""
    )
    
    # 체인 구성: 프롬프트 → LLM → 출력 파싱
    chain = prompt | llm | StrOutputParser()
    
    # 실행
    category = chain.invoke({"question": question}).strip()
    
    logger.info(f"분류 결과: {category}")
    return category
