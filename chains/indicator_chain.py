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
        temperature=0.4,
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
1. 경제지표의 현재 값과 의미를 초보 투자자도 이해할 수 있도록 쉽게 설명해주세요.
2. 질문과 관련된 경제지표가 주식 시장 전반 또는 특정 산업/기업에 미칠 수 있는 영향을 분석해주세요.
3. **반드시 다음 형식에 맞춰** 분석 내용을 작성해주세요:
    - 핵심 분석 내용을 먼저 간결하게 제시합니다.
    - 긍정적인 요인(기회)과 부정적인 요인(위험)을 명확히 구분하여 각각 '-'로 시작하는 목록 형태로 작성해주세요. (각 1~3개 항목)
    - 분석을 바탕으로 현재 경제 상황을 고려했을 때 적합한 투자 성향(공격적, 중립적, 안정적 중 하나)을 추천해주세요.

4. 답변은 한국어로 작성해주세요.
**[답변 형식]**
[핵심 분석]
(핵심 분석 내용 요약)

[긍정적 요인]
- (긍정적 영향 또는 기회 요인 1)
- (긍정적 영향 또는 기회 요인 2)

[부정적 요인]
- (부정적 영향 또는 위험 요인 1)
- (부정적 영향 또는 위험 요인 2)

[추천 투자 성향]
(공격적/중립적/안정적 중 택 1)
**[/답변 형식]**
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