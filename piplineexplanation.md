1. 사용자 질문 입력 (React → Spring Boot → FastAPI)
   ↓
2. 질문 분류 (Classifier Chain)
   - economic_indicator: 경제지표 관련
   - stock_price: 주가/재무 관련
   - analyst_report: 증권사 리포트 관련
   - general: 일반 투자 상담
   ↓
3. 카테고리별 처리
   - economic_indicator → DB 조회 + LLM 해석
   - stock_price → Yahoo Finance API + LLM 분석
   - analyst_report → RAG (ChromaDB 검색 + LLM 답변)
   - general → 직접 LLM 상담
   ↓
4. 답변 생성 (출처 포함)
   ↓
5. Spring Boot로 응답 반환
