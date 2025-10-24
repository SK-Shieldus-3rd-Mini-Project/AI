"""
주가 분석 체인 - API 데이터 조회 및 LLM 분석
"""
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.config import settings
from utils.logger import logger

def query_stock_analysis(question: str, stock_data: Dict[str, Any]):
    """주가 데이터를 기반으로 질문에 답변"""
    logger.info(f"주가 분석 질의: {question}")
    logger.debug(f"입력 주가 데이터: {stock_data}") # 입력 데이터 로깅
    
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
        openai_api_key=settings.openai_api_key
    )
    
    # 프롬프트 템플릿
    prompt = PromptTemplate(
        input_variables=["question", "stock_data_str"],
        template="""
당신은 주식 애널리스트입니다.
아래 주가 데이터를 기반으로 질문에 답변하세요.

제공된 주가 데이터:
{stock_data}

사용자 질문: {question}

1. 현재 주가, 등락률, 거래량 등 제공된 데이터를 명확히 언급하며 설명해주세요.
2. 데이터를 바탕으로 긍정적인 요인과 부정적인 요인을 분석해주세요. **반드시 다음 형식**을 따라 각 요인을 '-'로 시작하는 목록 형태로 작성해주세요. (각 1~3개 항목)
3. 종합적인 분석을 통해 이 종목 투자에 적합한 투자 성향(공격적, 중립적, 안정적 중 하나)을 추천해주세요.
4. 투자 시 주의사항을 언급하고, 답변 내용은 투자 추천이나 재무 자문이 아님을 명시해주세요.
5. 답변은 한국어로 작성해주세요.
**[답변 형식]**
[핵심 분석]
(핵심 분석 내용 요약)

[긍정적 요인]
- (긍정적 요인 1)
- (긍정적 요인 2)

[부정적 요인]
- (부정적 요인 1)
- (부정적 요인 2)

[추천 투자 성향]
(공격적/중립적/안정적 중 택 1)

[주의사항]
본 분석은 정보 제공 목적이며 투자 추천이나 재무 자문이 아닙니다. 투자 결정은 본인의 판단과 책임 하에 이루어져야 합니다.
**[/답변 형식]**:
"""
    )
    
    # 체인 구성
    chain = prompt | llm | StrOutputParser()
    
    # 주가 데이터를 문자열로 변환
    stock_str = "\n".join([f"- {k}: {v}" for k, v in stock_data.items() if v is not None]) # None 값은 제외
    if not stock_str: # 데이터가 비어있을 경우 대비
        stock_str = "제공된 주가 데이터 없음"
        logger.warning("stock_chain에 전달된 데이터가 비어있거나 유효하지 않습니다.")
    
    # 실행
    answer = chain.invoke({
        "question": question,
        "stock_data_str": stock_str
    })
    
    logger.info("주가 분석 답변 생성 완료")
    return answer
