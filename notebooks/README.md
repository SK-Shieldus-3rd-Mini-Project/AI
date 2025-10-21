

# 전봉준 AI 투자 어드바이저

## 프로젝트 개요
LangChain + RAG 기반 투자 상담 AI

## 설치 방법
\`\`\`bash
pip install -r requirements.txt
\`\`\`

## 환경 설정
1. `.env.example`을 `.env`로 복사
2. OpenAI API 키 입력

## 실행 방법
\`\`\`bash
python main.py
\`\`\`

## API 문서
http://localhost:8001/docs

## 프로젝트 구조
\`\`\`
AI/
├── chains/          # LangChain 체인
├── utils/           # 유틸리티
├── data/            # 데이터
├── notebooks/       # 테스트 노트북
└── main.py          # FastAPI 서버
\`\`\`
AI/
├── notebooks/                        #  실험/테스트
│   ├── 01_classifier_test.ipynb     # 질문 분류 테스트
│   ├── 02_rag_test.ipynb            # RAG 파이프라인 테스트
│   └── 03_full_pipeline_test.ipynb  # 전체 통합 테스트
├── chains/                           #  검증된 코드 (모듈화)
│   ├── classifier.py
│   ├── rag_chain.py
│   └── ...
├── utils/                            #  유틸리티
│   └── ...
└── main.py                           #  최종 API 서버