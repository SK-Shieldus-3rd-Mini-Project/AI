"""
RAG 체인 - 증권사 리포트 검색 및 답변 생성 (실제 구현)
"""
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.config import settings
from utils.db_client import get_vectorstore
from utils.logger import logger

def create_rag_chain(collection_name: str = "analyst_reports"):
    """
    RAG 체인 생성
    
    Args:
        collection_name: ChromaDB 컬렉션 이름
    
    Returns:
        RetrievalQA 체인
    """
    logger.info("RAG 체인 생성 시작")
    
    # LLM 초기화
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,  # 약간의 창의성 허용
        openai_api_key=settings.openai_api_key
    )
    
    # ★ ChromaDB에서 Vectorstore 로드 (임베딩된 리포트 문서들)
    vectorstore = get_vectorstore(collection_name=collection_name)
    
    # ★ Retriever 생성: 유사도 상위 3개 문서 반환
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}  # 관련 문서 3개 검색
    )
    
    # ★ RAG 프롬프트: 검색된 문서를 컨텍스트로 제공
    prompt_template = """
당신은 전문 투자 상담가입니다.
아래 증권사 리포트를 참고하여 질문에 답변하세요.

참고 문서:
{context}

질문: {question}

답변 지침:
1. 참고 문서의 내용을 기반으로 정확히 답변하세요
2. 출처를 명확히 밝히세요 (예: "NH투자증권 리포트에 따르면...")
3. 초보 투자자도 이해할 수 있도록 쉽게 설명하세요
4. 확실하지 않은 내용은 추측하지 마세요

답변:
"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    # ★ RAG Chain 생성: Retriever + LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 모든 문서를 하나의 컨텍스트로 결합
        retriever=retriever,
        return_source_documents=True,  # 출처 문서 반환
        chain_type_kwargs={"prompt": prompt}
    )
    
    logger.info("RAG 체인 생성 완료")
    return qa_chain

def query_rag(question: str, collection_name: str = "analyst_reports"):
    """
    RAG 체인 실행 (동기 방식)
    
    Args:
        question: 사용자 질문
        collection_name: 컬렉션 이름
    
    Returns:
        답변 및 출처 딕셔너리
    """
    logger.info(f"RAG 질의 시작: {question}")
    
    # RAG 체인 생성
    qa_chain = create_rag_chain(collection_name)
    
    # ★ 질의 실행: ChromaDB 검색 → LLM 답변 생성
    result = qa_chain({"query": question})
    
    # 결과 정리
    answer = result["result"]
    source_docs = result["source_documents"]
    
    # ★ 출처 정리: 메타데이터에서 증권사, 날짜 등 추출
    sources = []
    for doc in source_docs:
        sources.append({
            "title": doc.metadata.get("title", "Unknown"),
            "securities_firm": doc.metadata.get("securities_firm", "Unknown"),
            "date": doc.metadata.get("date", "Unknown"),
            "content": doc.page_content[:200]  # 일부만 표시
        })
    
    logger.info(f"RAG 답변 생성 완료: {len(answer)}자")
    
    return {
        "answer": answer,
        "sources": sources
    }