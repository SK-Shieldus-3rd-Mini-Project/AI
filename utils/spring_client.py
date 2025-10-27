
"""
Spring Boot 백엔드와 통신하는 클라이언트 모듈
경제지표 데이터를 Spring Boot의 MariaDB에서 가져옴
"""
import httpx
from typing import Dict, Optional
from utils.logger import logger
from utils.config import settings

class SpringBootClient:
    """Spring Boot API 클라이언트"""
    
    def __init__(self, base_url: str = "http://backend-svc:8080"):
        """
        초기화
        
        Args:
            base_url: Spring Boot 서버 주소
                     - 로컬: http://localhost:8080
                     - Kubernetes: http://backend-svc:8080
        """
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)  # 30초 타임아웃
    
    async def get_economic_indicators(self) -> Optional[Dict]:
        """
        경제지표 데이터 조회 (MariaDB)
        
        Returns:
            경제지표 딕셔너리 (기준금리, M2, 환율 등)
            실패 시 None
        """
        try:
            logger.info("Spring Boot에서 경제지표 조회 시작")
            
            # ★ Spring Boot에 이 API 엔드포인트를 구현해야 함 ★
            response = await self.client.get(
                f"{self.base_url}/api/indicators/latest"
            )
            response.raise_for_status()
            
            data = response.json()
            logger.info(f"경제지표 조회 성공: {data}")
            return data
            
        except httpx.HTTPStatusError as e:
            logger.error(f"경제지표 조회 HTTP 오류: {e.response.status_code}")
            return None
        except Exception as e:
            logger.error(f"경제지표 조회 실패: {str(e)}")
            return None
    
    async def close(self):
        """클라이언트 종료"""
        await self.client.aclose()

# 전역 클라이언트 인스턴스
spring_client = SpringBootClient()