# Dockerfile (in jeonbongjun-ai/ folder)

# Python 3.10 버전 사용 (예시)
FROM python:3.10-slim

WORKDIR /app

# requirements.txt 복사 및 설치
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 소스 코드 복사
COPY . .

# FastAPI 서버 포트 노출 (main.py에서 사용하는 포트와 일치)
EXPOSE 8000

# FastAPI 서버 실행 (uvicorn 사용)
# main:app -> main.py 파일의 app 변수를 의미
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]