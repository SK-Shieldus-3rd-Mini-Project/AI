import psycopg2
import yfinance as yf
import os
from langchain_openai import ChatOpenAI

class UserStocksChain:
    def __init__(self):
        self.db_params = {
            "host": os.getenv("DB_HOST", "localhost"),
            "database": os.getenv("DB_NAME", "investai"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "password"),
            "port": os.getenv("DB_PORT", "5432")
        }
        # LLM 초기화 (자연어 요약용)
        self.llm = ChatOpenAI(model="gpt-4", temperature=0.3)
    
    def get_user_stocks(self, user_id: str) -> dict:
        """
        사용자 보유 종목 조회 + 실시간 주가 정보 추가
        """
        try:
            # 1. DB에서 보유종목 조회
            conn = psycopg2.connect(**self.db_params)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT 
                    s.ticker_symbol,
                    s.stock_name,
                    p.quantity,
                    p.avg_purchase_price,
                    p.last_updated
                FROM user_portfolio p
                JOIN stock s ON p.stock_id = s.stock_id
                WHERE p.user_id = %s
                ORDER BY p.last_updated DESC
                LIMIT 10
            """, (user_id,))
            
            stocks = cur.fetchall()
            cur.close()
            conn.close()
            
            # 빈 결과 처리
            if not stocks:
                return {
                    "stocks": [],
                    "summary": "보유 중인 종목이 없습니다."
                }
            
            # 2. 각 종목의 실시간 정보 가져오기 (yfinance)
            detailed_stocks = []
            for row in stocks:
                ticker = row[0]
                stock_name = row[1]
                quantity = row[2]
                avg_price = float(row[3])
                last_updated = str(row[4])
                
                # 실시간 주가 조회
                stock_detail = self._get_stock_detail(ticker)
                
                detailed_stocks.append({
                    "ticker": ticker,
                    "name": stock_name,
                    "quantity": quantity,
                    "avg_price": avg_price,
                    "current_price": stock_detail.get("current_price", 0),
                    "change_percent": stock_detail.get("change_percent", 0),
                    "market_cap": stock_detail.get("market_cap", "N/A"),
                    "profit_loss": (stock_detail.get("current_price", 0) - avg_price) * quantity,
                    "last_updated": last_updated
                })
            
            # 3. LLM으로 자연어 요약 생성
            summary = self._generate_summary(detailed_stocks)
            
            return {
                "stocks": detailed_stocks,
                "summary": summary
            }
            
        except Exception as e:
            print(f"오류 발생: {str(e)}")
            return {
                "stocks": [],
                "summary": f"데이터 조회 실패: {str(e)}"
            }
    
    def _get_stock_detail(self, ticker: str) -> dict:
        """
        yfinance로 실시간 주가 정보 조회
        """
        try:
            # 한국 주식은 ticker 뒤에 .KS 또는 .KQ 붙이기
            if not ticker.endswith(('.KS', '.KQ')):
                ticker = f"{ticker}.KS"  # KOSPI 기본
            
            stock = yf.Ticker(ticker)
            info = stock.info
            
            return {
                "current_price": info.get("currentPrice", 0),
                "change_percent": info.get("regularMarketChangePercent", 0),
                "market_cap": info.get("marketCap", "N/A"),
                "volume": info.get("volume", 0)
            }
        except Exception as e:
            print(f"주가 조회 실패 ({ticker}): {str(e)}")
            return {
                "current_price": 0,
                "change_percent": 0,
                "market_cap": "N/A",
                "volume": 0
            }
    
    def _generate_summary(self, stocks: list) -> str:
        """
        LLM으로 보유종목 자연어 요약 생성
        """
        if not stocks:
            return "보유 중인 종목이 없습니다."
        
        stocks_text = "\n".join([
            f"- {s['name']} ({s['ticker']}): {s['quantity']}주 보유, "
            f"평균단가 {s['avg_price']:,.0f}원, 현재가 {s['current_price']:,.0f}원, "
            f"수익률 {s['change_percent']:.2f}%, 평가손익 {s['profit_loss']:,.0f}원"
            for s in stocks
        ])
        
        prompt = f"""
사용자가 보유한 종목 목록과 상세정보입니다:

{stocks_text}

위 종목들을 자연스럽게 요약하고, 간단한 투자 의견도 함께 제시해주세요.
"""
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except:
            return f"총 {len(stocks)}개 종목 보유 중: {', '.join([s['name'] for s in stocks])}"
