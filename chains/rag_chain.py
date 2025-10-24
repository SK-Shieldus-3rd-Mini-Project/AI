"""
RAG 체인 - 증권사 리포트 검색 및 답변 생성
"""
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.config import settings
from utils.db_client import get_vectorstore
from utils.logger import logger
from typing import Dict, List

def create_rag_chain(collection_name: str = "analyst_reports"):
    """
    RAG 체인 생성
    
    Args:
        collection_name: ChromaDB 컬렉션 이름
    
    Returns:
        RetrievalQA 체인
    """
    logger.info(f"RAG 체인 생성 시작 (Collection: {collection_name})")
    
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,  # 약간의 창의성 허용
        openai_api_key=settings.openai_api_key
    )
    
    # Vectorstore Retriever (유사도 검색)
    vectorstore = get_vectorstore(collection_name=collection_name)
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # 유사도 기반 검색
        search_kwargs={"k": 3}  # 상위 3개 문서 반환
    )
    
    # RAG 프롬프트 (출처 명확히 표시)
    prompt_template = """
당신은 전문 투자 상담가입니다.
아래 제공된 증권사 리포트 내용을 참고하여 사용자의 질문에 답변하세요.

참고 문서:
{context}

사용자 질문: {question}

[답변 지침]
1. 참고 문서의 내용을 기반으로 질문에 대해 정확하고 상세하게 답변하세요.
2. 답변 내용 중 참고 문서에서 근거를 찾을 수 있는 부분은 출처(증권사, 날짜 등 메타데이터)를 명시해주세요. (예: "NH투자증권(2025-10-15) 리포트에 따르면...")
3. **반드시 다음 형식에 맞춰** 분석 내용을 작성해주세요:
    - 핵심 분석 내용을 먼저 간결하게 제시합니다.
    - 긍정적인 요인(기회)과 부정적인 요인(위험)을 명확히 구분하여 각각 '-'로 시작하는 목록 형태로 작성해주세요. (각 1~3개 항목)
    - 분석을 바탕으로 이 정보에 기반한 투자 성향(공격적, 중립적, 안정적 중 하나)을 추천해주세요.
4. 초보 투자자도 이해할 수 있도록 쉬운 용어를 사용해주세요.
5. 확실하지 않거나 문서에 없는 내용은 추측하지 마세요.
6. 답변은 한국어로 작성해주세요.

**[답변 형식]**
[핵심 분석]
(핵심 분석 내용 요약)

[긍정적 요인]
- (긍정적 요인 1) (출처: 증권사(날짜))
- (긍정적 요인 2)

[부정적 요인]
- (부정적 요인 1)
- (부정적 요인 2) (출처: 증권사(날짜))

[추천 투자 성향]
(공격적/중립적/안정적 중 택 1)
**[/답변 형식]**

답변:
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    # RAG Chain 생성
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 모든 문서를 하나의 컨텍스트로 결합
        retriever=retriever,
        return_source_documents=True,  # 출처 문서 반환
        chain_type_kwargs={"prompt": prompt}
    )
    
    logger.info("RAG 체인 생성 완료")
    return qa_chain

def query_rag(question: str, collection_name: str = "analyst_reports") -> Dict[str, Any]:
    """
    RAG 체인 실행 및 결과 반환

    Args:
        question: 사용자 질문
        collection_name: 컬렉션 이름

    Returns:
        Dict: 'answer'(str)와 'sources'(List[Dict])를 포함하는 딕셔너리
              sources의 각 Dict는 'title', 'securities_firm', 'date', 'content' 키를 가짐
    """
    logger.info(f"RAG 질의 시작: {question} (Collection: {collection_name})")

    # RAG 체인 생성
    qa_chain = create_rag_chain(collection_name)

    # 체인 실행 (invoke 사용 권장)
    try:
        # result = qa_chain({"query": question}) # 이전 방식 (DeprecationWarning 발생 가능)
        result = qa_chain.invoke({"query": question}) # 최신 방식
    except Exception as e:
        logger.error(f"RAG 체인 실행 중 오류 발생: {e}", exc_info=True)
        return {"answer": "죄송합니다. RAG 답변 생성 중 오류가 발생했습니다.", "sources": []}

    # 결과 정리
    answer = result.get("result", "답변을 생성하지 못했습니다.") # 'result' 키 확인
    source_docs = result.get("source_documents", []) # 'source_documents' 키 확인

    # 출처 정보 정리 (메타데이터 + 내용 일부)
    sources = []
    if source_docs: # 출처 문서가 있을 경우에만 처리
        for doc in source_docs:
            metadata = doc.metadata if hasattr(doc, 'metadata') else {}
            sources.append({
                "title": metadata.get("title", "제목 없음"),
                "securities_firm": metadata.get("securities_firm", "출처 불명"),
                "date": metadata.get("date", "날짜 불명"),
                "content": doc.page_content[:200] + "..." if hasattr(doc, 'page_content') else "" # 내용 일부 (200자)
            })

    logger.info(f"RAG 답변 생성 완료: 답변 길이={len(answer)}, 출처 개수={len(sources)}")

    # 딕셔너리 형태로 결과 반환
    return {
        "answer": answer,
        "sources": sources
    }