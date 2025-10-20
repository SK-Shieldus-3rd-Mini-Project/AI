from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from utils.config import settings

app = FastAPI(
    title="전봉준 AI API",
    description="투자 및 경제 인사이트 AI 챗봇 API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    session_id: str
    question: str

class QueryResponse(BaseModel):
    session_id: str
    question: str
    answer: str
    category: str
    timestamp: str

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "service": "InvestAI Core",
        "version": "1.0.0"
    }

@app.post("/api/ai/query", response_model=QueryResponse)
async def query_ai(request: QueryRequest):
    try:
        # MVP 더미 응답
        answer = f"[MVP] {request.question}에 대한 답변입니다."
        
        response = QueryResponse(
            session_id=request.session_id,
            question=request.question,
            answer=answer,
            category="general",
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
