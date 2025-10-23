from langchain_openai import ChatOpenAI
from typing import List, Dict
import yfinance as yf
import psycopg2
class UserStocksChain:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def get_user_stocks(self, user_id: str) -> Dict:
        """
        사용자 최근 추가 종목 목록과 상세정보를 조회
        """
        # 1. DB에서 사용자 종목 조회 (실제로는 DB 쿼리 또는 API 호출)
        # 예시 데이터 (실제로는 DB/API로 대체)
        stocks = self._fetch_stocks_from_db(user_id)
        
        # 2. 각 종목의 상세정보 가져오기 (실시간 API or DB)
        detailed_stocks = []
        for stock in stocks:
            detail = self._get_stock_detail(stock['stock_code'])
            detailed_stocks.append({
                "name": stock['stock_name'],
                "code": stock['stock_code'],
                "added_date": stock['added_date'],
                "current_price": detail['price'],
                "change": detail['change'],
                "volume": detail['volume'],
                "market_cap": detail['market_cap']
            })
        
        # 3. LLM으로 자연어 요약 생성
        summary = self._generate_summary(detailed_stocks)
        
        return {
            "stocks": detailed_stocks,
            "summary": summary
        }
    
    def _fetch_stocks_from_db(self, user_id: str) -> List[Dict]:
        """
        실제로는 DB 쿼리 (예: PostgreSQL, MySQL)
        """
        # 예시 더미 데이터
        return [
            {"stock_code": "005930", "stock_name": "삼성전자", "added_date": "2025-10-20"},
            {"stock_code": "035420", "stock_name": "네이버", "added_date": "2025-10-21"},
            {"stock_code": "035720", "stock_name": "카카오", "added_date": "2025-10-22"}
        ]
    
    def _get_stock_detail(self, stock_code: str) -> Dict:
        # 실시간 주가 API 예시 (한국 종목 코드)
        ticker = yf.Ticker(f"{stock_code}.KS")
        info = ticker.info
        return {
            "price": info.get("currentPrice", 0),
            "change": info.get("regularMarketChangePercent", 0),
            "volume": info.get("volume", 0),
            "market_cap": info.get("marketCap", 0)
        }
    
    def _generate_summary(self, stocks: List[Dict]) -> str:
        """
        LLM으로 자연어 요약
        """
        stocks_text = "\n".join([
            f"- {s['name']} ({s['code']}): 현재가 {s['current_price']}원, 전일대비 {s['change']}, 시가총액 {s['market_cap']}"
            for s in stocks
        ])
        
        prompt = f"""
사용자가 최근 추가한 종목 목록과 상세정보입니다:

{stocks_text}

위 종목들을 자연스럽게 요약하고, 간단한 투자 의견도 함께 제시해주세요.
"""
        response = self.llm.invoke(prompt)
        return response.content

class UserStocksChain:
    def __init__(self):
        # 필요시 DB 연결 정보 환경 변수에서 불러오게 변경
        self.db_params = dict(
            host="localhost",
            database="your_db",
            user="your_user",
            password="your_password"
        )

    def get_user_stocks(self, user_id: str) -> dict:
        conn = psycopg2.connect(**self.db_params)
        cur = conn.cursor()
        cur.execute("""
            SELECT stock_code, stock_name, added_date 
            FROM user_stocks 
            WHERE user_id = %s 
            ORDER BY added_date DESC 
            LIMIT 5
        """, (user_id,))
        stocks = cur.fetchall()
        conn.close()

        # 상세정보/실시간 시세는 API 연동 또는 추가 쿼리로 가져와야 함
        stocks_list = [{
            "code": row[0],
            "name": row[1],
            "added_date": str(row[2])
            # "current_price": ... (API, DB에서 추가로 가져올 수 있음)
        } for row in stocks]
        
        # 자연어 요약은 LLM 활용 또는 간단하게 text 생성
        summary = f"최근 추가된 종목: {', '.join([s['name'] for s in stocks_list])}"

        return {
            "stocks": stocks_list,
            "summary": summary
        }
