

from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import settings
from utils.logger import logger

def query_general_advice(question: str) -> str: # 반환 타입 명시
    """
    일반적인 투자 상담 질문에 답변 (RAG나 외부 데이터 없이 LLM 자체 지식 활용)

    Args:
        question: 사용자 질문

    Returns:
        str: LLM이 생성한 답변 문자열
    """
    logger.info(f"일반 상담 질의 시작: {question}")

    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.5,  # 일반 상담은 약간 더 창의적인 답변 허용
        openai_api_key=settings.openai_api_key
    )

    # --- 프롬프트 수정: 긍정/부정/성향 분석 지침 추가 ---
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
당신은 친절하고 지식이 풍부한 투자 상담 전문가입니다.
사용자의 일반적인 투자 관련 질문에 대해 초보자도 이해할 수 있도록 쉽고 정확하게 답변해주세요.

[사용자 질문]
{question}

[답변 지침]
1. 복잡한 금융 용어는 쉽게 풀어서 설명하고, 필요한 경우 구체적인 예시를 들어주세요.
2. 질문의 내용이 특정 투자 전략이나 시장 상황 분석을 포함하는 경우, **다음 형식에 맞춰** 긍정적 측면과 부정적 측면(위험)을 함께 고려하여 균형 잡힌 답변을 제공해주세요.
    - 핵심 내용을 먼저 요약합니다.
    - 긍정적 측면과 부정적 측면을 각각 '-'로 시작하는 목록 형태로 작성합니다. (각 1~3개 항목)
    - 가능하다면 해당 전략이나 상황에 적합한 투자 성향(공격적, 중립적, 안정적)을 언급해주세요.
3. 답변 내용은 일반적인 정보 제공 목적이며, 법적 또는 재무적 자문으로 해석되어서는 안 됨을 명시해주세요.
4. 답변은 한국어로 작성해주세요.

**[분석 포함 시 답변 형식 예시]**
[핵심 요약]
(핵심 내용 요약)

[긍정적 측면]
- (긍정적 측면 1)
- (긍정적 측면 2)

[부정적 측면]
- (부정적 측면 1)
- (부정적 측면 2)

[적합 투자 성향]
(공격적/중립적/안정적 또는 해당 없음)

[주의사항]
본 답변은 일반적인 정보 제공 목적이며 투자 자문이 아닙니다. 모든 투자 결정은 본인의 판단과 책임 하에 신중히 이루어져야 합니다.
**[/분석 포함 시 답변 형식 예시]**

답변:
"""
    )

    # 체인 구성
    chain = prompt | llm | StrOutputParser()

    # 체인 실행
    answer = chain.invoke({"question": question})

    logger.info("일반 상담 답변 생성 완료")
    # LLM이 생성한 전체 문자열 반환 (main.py에서 파싱 예정)
    return answer