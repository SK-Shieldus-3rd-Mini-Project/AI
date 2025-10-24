"""
FastAPI 메인 서버
모든 체인을 통합하여 Spring Boot와 연동
"""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Dict, Optional, Any

# 설정 및 유틸
from utils.config import settings
from utils.logger import logger

# 체인들
from chains.classifier import classify_question
from chains.rag_chain import query_rag
from chains.indicator_chain import query_economic_indicator
from chains.stock_chain import query_stock_analysis
from chains.general_chain import query_general_advice
from chains.sentiment_analyzer import SentimentAnalyzer
from chains.user_stocks_chain import UserStocksChain

# pykrx API 추가
from pykrx import stock
import pandas as pd
import re


# 인스턴스 생성
sentiment_analyzer = SentimentAnalyzer()
user_stocks_chain = UserStocksChain()

# FastAPI 앱 초기화
app = FastAPI(
    title="전봉준 AI 투자 어드바이저 API",
    description="CHATBOT 기반 투자 상담 API",
    version="1.0.0"
)

# CORS 설정 (React, Spring Boot 연동)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 배포 시 특정 도메인만 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 요청/응답 모델 =====
class IndicatorData(BaseModel):
    """백엔드가 전달할 경제 지표 데이터 형식 (최종 확정된 필드)"""
    name: str 
    value: str 
    date: str = None 
    unit: str = None 
class StockData(BaseModel):
    """백엔드가 전달할 주가 데이터 형식"""
    ticker: str # 종목 코드 
    current_price: Optional[float] = None # 현재가
    change_percent: Optional[float] = None # 등락률
    name: Optional[str] = None # 종목명 등 추가    
class UserRequest(BaseModel):
    """사용자 정보 요청 모델"""
    user_id: str

class QueryRequest(BaseModel):
    """질문 요청 모델"""
    session_id: str = Field(..., alias="sessionId")  # 세션 ID (사용자 식별)
    question: str    # 사용자 질문
    indicator_data: Optional[List[IndicatorData]] = Field(default=None, alias="indicatorData")
    stock_data: Optional[StockData] = Field(default=None, alias="stockData")
    # Pydantic 설정: alias 사용 및 예외 필드 허용 여부 등 설정 가능
    class Config:
        populate_by_name = True # alias를 JSON 필드명으로 인식
    
    # --- 2. 출력 모델 (스크린샷 + 신규기능 반영, 백엔드 AiResponseDto 와 매핑) ---
class EconomicDataUsed(BaseModel):
     """AI 답변에 사용된 경제 데이터"""
     name: str
     value: str
class SourceCitation(BaseModel):
     """AI 답변의 근거 출처 (뉴스, 웹사이트 등 - RAG 외)"""
     title: str # 출처 제목 (예: "한국은행 경제통계시스템")
     url: Optional[str] = None # 출처 URL (있을 경우)
class RelatedReport(BaseModel):
     """RAG 검색 결과로 나온 관련 리포트 정보"""
     # rag_chain.py에서 반환하는 source dict의 키와 일치시키는 것이 좋음
     title: str # 리포트 제목
     securities_firm: Optional[str] = None # 증권사
     date: Optional[str] = None # 리포트 날짜   
class ResponseDetails(BaseModel):
     """AI 답변의 상세 정보 구조"""
     # Field의 alias를 사용하여 Python(snake_case)과 JSON(camelCase) 변환
     economic_data_used: Optional[List[EconomicDataUsed]] = Field(default=None, alias="economicDataUsed")
     source_citations: Optional[List[SourceCitation]] = Field(default=None, alias="sourceCitations")
     related_reports: Optional[List[RelatedReport]] = Field(default=None, alias="relatedReports")
     # --- 긍정/부정 분석 필드 ---
     positive_points: Optional[List[str]] = Field(default=None, alias="positivePoints") # 긍정 요인 목록
     negative_points: Optional[List[str]] = Field(default=None, alias="negativePoints") # 부정 요인 목록
     suggested_propensity: Optional[str] = Field(default=None, alias="suggestedPropensity") # 추천 투자 성향 (예: "중립적")  
class QueryResponse(BaseModel):
    """질문 응답 모델 (Spring Boot AiResponseDto 와 매핑)"""
    session_id: str = Field(..., alias="sessionId") # 세션 ID
    question: str # 원본 사용자 질문
    answer: str # LLM이 생성한 주 답변 내용 (긍정/부정 분석 포함 가능)
    category: str # 분류된 질문 카테고리
    timestamp: str # 응답 생성 시간 (ISO 형식)
    details: Optional[ResponseDetails] = None # 상세 정보 (내용이 있을 때만 포함)

    class Config:
        populate_by_name = True # alias 허용

class StocksResponse(BaseModel):
    """보유 종목 응답 모델"""
    success: bool
    user_id: str
    data: Dict
# --- 유틸리티 함수: LLM 답변에서 긍정/부정/성향 파싱 ---
def parse_analysis_from_answer(answer_content: str) -> Optional[Dict[str, Any]]:
    """LLM 답변 문자열에서 특정 키워드('[긍정적 요인]' 등)를 기준으로 분석 결과 파싱"""
    results = {
        "positive_points": None,
        "negative_points": None,
        "suggested_propensity": None
    }
    # LLM이 "[키워드]\n- 내용1\n- 내용2" 형식으로 출력한다고 가정
    keywords_map = {
        "[긍정적 요인]": "positive_points",
        "[부정적 요인]": "negative_points",
        "[추천 투자 성향]": "suggested_propensity"
    }
    sections: Dict[str, str] = {}
    lines = answer_content.splitlines()
    current_keyword = None
    buffer = []

    # 답변 라인을 순회하며 섹션 분리
    for line in lines:
        line_strip = line.strip()
        found_keyword = None
        for kw in keywords_map:
            if line_strip.startswith(kw):
                found_keyword = kw
                break
        if found_keyword:
            if current_keyword: # 이전 섹션 내용 저장
                sections[current_keyword] = "\n".join(buffer).strip()
            current_keyword = found_keyword
            # 키워드 자체를 제외한 첫 줄 내용 추가 (있을 경우)
            buffer = [line_strip[len(found_keyword):].strip()]
        elif current_keyword: # 현재 섹션 내용 계속 추가
            buffer.append(line)
    if current_keyword: # 마지막 섹션 내용 저장
        sections[current_keyword] = "\n".join(buffer).strip()

    # 파싱된 섹션에서 실제 데이터 추출
    if "[긍정적 요인]" in sections:
        # '-'로 시작하는 목록 추출
        points_text = sections["[긍정적 요인]"]
        results["positive_points"] = [p.strip("- ").strip() for p in points_text.split('\n') if p.strip().startswith("- ")]
    if "[부정적 요인]" in sections:
        points_text = sections["[부정적 요인]"]
        results["negative_points"] = [p.strip("- ").strip() for p in points_text.split('\n') if p.strip().startswith("- ")]
    if "[추천 투자 성향]" in sections:
        # 추천 성향은 첫 줄 내용 사용
        results["suggested_propensity"] = sections["[추천 투자 성향]"].split('\n')[0].strip()

    # 결과가 하나라도 있으면 딕셔너리 반환, 없으면 None 반환
    if any(results.values()):
        return results
    else:
        logger.debug("LLM 답변에서 분석 결과 키워드를 찾지 못했습니다.")
        return None







# ===== API 엔드포인트 =====

@app.get("/health")
async def health_check():
    """헬스 체크 - 서버 상태 확인"""
    return {
        "status": "ok",
        "service": "InvestAI Core",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/ai/my-stocks", response_model=StocksResponse)
async def get_my_stocks(request: UserRequest):
    """
    사용자 보유 종목 조회 API
    - DB에서 USER_PORTFOLIO 기반 보유 종목 조회
    - yfinance로 실시간 주가 정보 추가
    - LLM으로 자연어 요약 생성
    """
    try:
        logger.info(f"[{request.user_id}] 보유 종목 조회 요청")
        
        # user_stocks_chain에서 데이터 조회
        result = user_stocks_chain.get_user_stocks(request.user_id)
        
        logger.info(f"[{request.user_id}] 보유 종목 조회 완료")
        
        return {
            "success": True,
            "user_id": request.user_id,
            "data": {
                "stocks": result["stocks"],
                "summary": result["summary"]
            }
        }
    except Exception as e:
        logger.error(f"[{request.user_id}] 보유 종목 조회 오류: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/query", response_model=QueryResponse, response_model_exclude_none=True) # None 필드 제외 옵션 추가
async def query_ai(request: QueryRequest): # 수정된 QueryRequest 사용
    """ AI 질문 처리 메인 엔드포인트 """
    try:
        # 로그: 수신된 요청 데이터 확인 (디버깅 시 유용)
        logger.info(f"[{request.session_id}] 질문 수신: {request.question}")
        # 입력 데이터 상세 로깅 (개인 정보 포함될 수 있으므로 주의)
        # logger.debug(f"[{request.session_id}] 수신 데이터: {request.model_dump_json(exclude_unset=True)}")

        # 1. 질문 분류 (기존 유지)
        category = classify_question(request.question)
        logger.info(f"[{request.session_id}] 분류 결과: {category}")

        # --- 응답 구성 요소 초기화 ---
        answer_content = ""           # LLM의 전체 텍스트 답변
        sources_from_rag: List[Dict] = [] # RAG 결과 (title, securities_firm, date, content 등)
        economic_data_used_list: List[EconomicDataUsed] = [] # 사용된 경제 데이터 리스트
        parsed_analysis: Optional[Dict[str, Any]] = None # 파싱된 긍정/부정/성향

        # 2. 카테고리별 처리 로직
        if category == "analyst_report":
            # RAG 체인 호출
            rag_result = query_rag(request.question)
            answer_content = rag_result.get("answer", "리포트 기반 답변 생성 중 오류가 발생했습니다.")
            sources_from_rag = rag_result.get("sources", []) # 리스트 반환 가정
            # RAG 답변에서도 긍정/부정 분석 시도 (RAG 프롬프트 수정 필요)
            parsed_analysis = parse_analysis_from_answer(answer_content)
            logger.debug(f"[{request.session_id}] RAG 결과: {len(sources_from_rag)}개 출처 찾음")

        elif category == "economic_indicator":
            # 백엔드에서 전달된 경제 지표 데이터 확인
            if request.indicator_data:
                # --- 실제 indicator_data 사용 ---
                # indicator_chain 실행 (실제 데이터를 인자로 전달)
                answer_content = query_economic_indicator(request.question, request.indicator_data)
                # 입력 데이터를 응답 details에 포함시키기 위해 변환
                economic_data_used_list = [
                    EconomicDataUsed(name=d.name, value=d.value) for d in request.indicator_data
                ]
                # 답변 내용에서 분석 결과 파싱
                parsed_analysis = parse_analysis_from_answer(answer_content)
                logger.debug(f"[{request.session_id}] 경제 지표 데이터 활용: {len(economic_data_used_list)}개")
            else:
                # 데이터가 없을 경우, 일반 상담 체인으로 처리하거나 메시지 반환
                answer_content = "관련 경제 지표 데이터가 필요합니다. 다시 질문해주세요."
                logger.warning(f"[{request.session_id}] economic_indicator 질문에 indicator_data 누락")

        elif category == "stock_price":
            # 백엔드에서 전달된 주가 데이터 확인
            if request.stock_data:
                # --- 실제 stock_data 사용 ---
                # stock_chain 실행 (실제 데이터를 dict 형태로 전달)
                # Pydantic 모델을 dict로 변환 시 exclude_none=True로 None 필드 제외
                stock_data_dict = request.stock_data.model_dump(exclude_none=True)
                answer_content = query_stock_analysis(request.question, stock_data_dict)
                # 답변 내용에서 분석 결과 파싱
                parsed_analysis = parse_analysis_from_answer(answer_content)
                logger.debug(f"[{request.session_id}] 주가 데이터 활용: {request.stock_data.ticker}")
            else:
                # 데이터가 없을 경우 처리
                answer_content = "해당 종목의 주가 정보가 필요합니다. 다시 질문해주세요."
                logger.warning(f"[{request.session_id}] stock_price 질문에 stock_data 누락")

        else:  # "general" 또는 예상치 못한 카테고리
            # 일반 상담 체인 호출
            answer_content = query_general_advice(request.question)
            # 일반 답변에서도 분석 결과 파싱 시도
            parsed_analysis = parse_analysis_from_answer(answer_content)
            logger.debug(f"[{request.session_id}] 일반 상담 처리")


        # --- 3. 최종 응답 객체 생성 ---
        # RAG 검색 결과를 RelatedReport 리스트로 변환 (None 안전 처리 추가)
        related_reports_list = [
            RelatedReport(
                title=src.get("title", "제목 없음"), # title 필드는 필수라고 가정
                securities_firm=src.get("securities_firm"), # Optional 필드는 None 가능
                date=src.get("date")
            ) for src in sources_from_rag
        ] if sources_from_rag else None

        # 상세 정보 객체 생성
        response_details = ResponseDetails(
            # 값이 있는 경우에만 리스트 생성, 아니면 None
            economicDataUsed=economic_data_used_list if economic_data_used_list else None,
            relatedReports=related_reports_list,
            # 파싱된 분석 결과 매핑 (None 안전 처리 추가)
            positivePoints=parsed_analysis.get("positive_points") if parsed_analysis else None,
            negativePoints=parsed_analysis.get("negative_points") if parsed_analysis else None,
            suggestedPropensity=parsed_analysis.get("suggested_propensity") if parsed_analysis else None,
            # sourceCitations = ... # 필요시 경제 지표 출처(예: 한국은행) 등 여기에 추가
        )

        # 최종 응답 생성 (QueryResponse 모델 사용)
        response = QueryResponse(
            sessionId=request.session_id, # 입력받은 sessionId 사용 (alias 적용됨)
            question=request.question,
            answer=answer_content, # LLM의 전체 답변
            category=category,
            timestamp=datetime.now().isoformat(),
            # details 필드는 내부 필드 중 하나라도 값이 있을 때만 포함
            details=response_details if any(value is not None for value in vars(response_details).values()) else None
        )

        logger.info(f"[{request.session_id}] 응답 생성 완료")
        # Pydantic 모델 객체를 반환 (FastAPI가 JSON 변환 처리, alias 및 exclude_none 적용)
        return response

    except HTTPException as http_exc:
        # FastAPI가 발생시킨 HTTP 예외는 그대로 전달
        raise http_exc
    except Exception as e:
        # 그 외 예상치 못한 내부 오류 로깅 및 일반 오류 응답
        logger.error(f"[{request.session_id}] AI 처리 중 심각한 오류 발생: {str(e)}", exc_info=True) # 스택 트레이스 포함 로깅
        raise HTTPException(status_code=500, detail="AI 답변 생성 중 내부 서버 오류가 발생했습니다.")

# ===== 서버 실행 (기존 유지) =====
if __name__ == "__main__":
    import uvicorn
    logger.info(f"AI Core 서버 시작 (Host: {settings.host}, Port: {settings.port}, Debug: {settings.debug})")
    uvicorn.run(
        "main:app", # FastAPI 앱 객체 지정 (파일명:앱객체명)
        host=settings.host,
        port=settings.port,
        reload=settings.debug # 코드 변경 시 자동 재시작 (개발 환경에서 유용)
    )
