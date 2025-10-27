"""
RAG 체인 - 증권사 리포트 검색 및 답변 생성 (타입 안정성 강화)
"""
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from utils.config import settings
from utils.db_client import get_vectorstore
from utils.logger import logger
from typing import Dict, Any, List

def create_rag_chain(collection_name: str = "analyst_reports") -> RetrievalQA:
    """
    RAG 체인 생성
    
    Args:
        collection_name: ChromaDB 컬렉션 이름
    
    Returns:
        RetrievalQA 체인
    """
    logger.info("RAG 체인 생성 시작")
    
    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=0.3,
        openai_api_key=settings.openai_api_key
    )
    
    vectorstore = get_vectorstore(collection_name=collection_name)
    
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
    
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
5. 여러 증권사의 의견이 다르면 모두 소개하세요

답변:
"""
    
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    
    logger.info("RAG 체인 생성 완료")
    return qa_chain

def query_rag(question: str, collection_name: str = "analyst_reports") -> Dict[str, Any]:
    """
    RAG 체인 실행 (타입 안정성 강화)
    
    Args:
        question: 사용자 질문
        collection_name: 컬렉션 이름
    
    Returns:
        {
            "answer": str,
            "sources": List[Dict[str, str]]
        }
    """
    logger.info(f"RAG 질의 시작: {question}")
    
    try:
        qa_chain = create_rag_chain(collection_name)
        
        result = qa_chain({"query": question})
        
        answer: str = result["result"]
        source_docs = result["source_documents"]
        
        sources: List[Dict[str, str]] = []
        for doc in source_docs:
            sources.append({
                "title": doc.metadata.get("title", "Unknown"),
                "securities_firm": doc.metadata.get("securities_firm", "Unknown"),
                "date": doc.metadata.get("date", "Unknown"),
                "content": doc.page_content[:200]
            })
        
        logger.info(f"RAG 답변 생성 완료: {len(answer)}자, 출처 {len(sources)}개")
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    except Exception as e:
        logger.error(f"RAG 체인 실행 실패: {e}", exc_info=True)
        return {
            "answer": f"증권사 리포트 검색 중 오류가 발생했습니다: {str(e)}",
            "sources": []
        }