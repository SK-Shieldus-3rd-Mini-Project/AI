AI/
├── data/
│   ├── reports/                  # 증권사 리포트 PDF
│   ├── news/                     # 뉴스 데이터
│   └── indicators/               # 경제지표 CSV
├── embeddings/
│   └── chromadb/                 # 벡터 DB
├── chains/
│   ├── __init__.py
│   ├── classifier.py             # 질문 분류
│   ├── rag_chain.py              # RAG 체인
│   ├── indicator_chain.py        # 경제지표 체인
│   ├── stock_chain.py            # 주가 분석 체인
│   └── general_chain.py          # 일반 상담 체인
├── utils/
│   ├── __init__.py
│   ├── config.py                 # 환경 설정
│   ├── logger.py                 # 로깅
│   ├── embedder.py               # 임베딩
│   ├── db_client.py              # ChromaDB
│   ├── data_loader.py            # 문서 로딩
│   └── text_splitter.py          # 텍스트 분할
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   └── 02_rag_test.ipynb
├── main.py                       # FastAPI 서버
├── requirements.txt
├── .env
└── README.md





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


[사용자 질문] (React → Spring Boot)
    ↓
[FastAPI: POST /api/ai/query]
    ↓
[질문 분류] (chains/classifier.py)
    ↓
┌─────────┬────────────┬─────────────┬──────────┐
│economic │stock_price │analyst_     │general   │
│indicator│            │report       │          │
└────┬────┴─────┬──────┴──────┬──────┴────┬─────┘
     │          │             │           │
     ↓          ↓             ↓           ↓
  [DB조회]  [API조회]    [RAG검색]   [LLM직접]
  indicator  stock_      rag_chain   general_
  _chain.py  chain.py    .py         chain.py
     │          │             │           │
     └──────────┴─────────────┴───────────┘
                    ↓
            [답변 + 출처 생성]
                    ↓
        [JSON 응답 → Spring Boot]
