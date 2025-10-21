AI/
├── notebooks/                        # 🧪 실험/테스트
│   ├── 01_classifier_test.ipynb     # 질문 분류 테스트
│   ├── 02_rag_test.ipynb            # RAG 파이프라인 테스트
│   └── 03_full_pipeline_test.ipynb  # 전체 통합 테스트
├── chains/                           # ✅ 검증된 코드 (모듈화)
│   ├── classifier.py
│   ├── rag_chain.py
│   └── ...
├── utils/                            # ✅ 유틸리티
│   └── ...
└── main.py                           # 🚀 최종 API 서버
