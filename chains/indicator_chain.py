"""
경제지표 체인 - Spring Boot DB 데이터 조회 및 LLM 해석
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import settings
from utils.logger import logger
from utils.spring_client import spring_client

async def query_economic_indicator(question: str):
    """
    경제지표 데이터를 기반으로 질문에 답변 (비동기)
    
    Args:
        question: 사용자 질문
    
    Returns:
        답변 문자열
    """
    logger.info(f"경제지표 질의: {question}")
    
    # ★ Spring Boot에서 MariaDB 경제지표 데이터 조회
    indicator_data = await spring_client.get_economic_indicators()
    
    if not indicator_data:
        return "죄송합니다. 경제지표 데이터를 조회할 수 없습니다."
    
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
        openai_api_key=settings.openai_api_key
    )
    
    # ★ 프롬프트: 경제지표 데이터를 컨텍스트로 제공
    prompt = PromptTemplate(
        input_variables=["question", "indicators"],
        template="""
당신은 경제 전문가입니다.
아래 경제지표 데이터를 기반으로 질문에 답변하세요.

현재 경제지표:
{indicators}

질문: {question}

답변 지침:
1. 경제지표의 의미를 초보자도 이해할 수 있게 설명하세요
2. 주식시장에 미치는 영향을 구체적으로 설명하세요
3. 특정 업종이나 기업에 미치는 영향도 언급하세요

답변:
"""
    )
    
    # 체인 구성
    chain = prompt | llm | StrOutputParser()
    
    # ★ 경제지표를 문자열로 변환
    indicators_str = "\n".join([f"- {k}: {v}" for k, v in indicator_data.items()])
    
    # 실행
    answer = chain.invoke({
        "question": question,
        "indicators": indicators_str
    })
    
    logger.info("경제지표 답변 생성 완료")
    return answer