"""
일반 투자 상담 체인 - 직접 LLM 답변
"""
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import settings
from utils.logger import logger

def query_general_advice(question: str):
    """
    일반적인 투자 상담 질문에 답변
    
    Args:
        question: 사용자 질문
    
    Returns:
        답변 문자열
    """
    logger.info(f"일반 상담 질의: {question}")
    
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.5,  # 조금 더 창의적 답변
        openai_api_key=settings.openai_api_key
    )
    
    # 프롬프트 템플릿
    prompt = PromptTemplate(
        input_variables=["question"],
        template="""
당신은 친절한 투자 상담 전문가입니다.
초보 투자자가 이해할 수 있도록 쉽고 정확하게 답변하세요.

질문: {question}

답변 지침:
1. 복잡한 용어는 쉽게 풀어서 설명하세요
2. 구체적인 예시를 들어주세요
3. 투자 위험에 대해서도 언급하세요
4. 법적/재무적 조언이 아님을 명시하세요

답변:
"""
    )
    
    # 체인 구성
    chain = prompt | llm | StrOutputParser()
    
    # 실행
    answer = chain.invoke({"question": question})
    
    logger.info("일반 상담 답변 생성 완료")
    return answer
