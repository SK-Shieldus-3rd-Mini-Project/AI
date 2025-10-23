from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os

class SentimentAnalyzer:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.3,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # 긍정 의견 추출 프롬프트
        self.positive_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 투자 분석 전문가입니다. 
주어진 정보에서 **긍정적인 투자 의견, 강점, 성장 가능성, 긍정적 전망만** 추출하세요.
근거가 부족하면 "긍정적 의견 정보 부족"이라고 답하세요."""),
            ("user", "질문: {question}\n\n참고 문서:\n{context}\n\n긍정적 의견만 요약:")
        ])
        
        # 부정 의견 추출 프롬프트
        self.negative_prompt = ChatPromptTemplate.from_messages([
            ("system", """당신은 투자 분석 전문가입니다.
주어진 정보에서 **부정적인 투자 의견, 리스크, 약점, 하방압력, 부정적 전망만** 추출하세요.
근거가 부족하면 "부정적 의견 정보 부족"이라고 답하세요."""),
            ("user", "질문: {question}\n\n참고 문서:\n{context}\n\n부정적 의견만 요약:")
        ])
    
    def analyze(self, question: str, context: str) -> dict:
        """긍정/부정 의견을 분리 추출"""
        try:
            # 긍정 의견 추출
            positive_chain = self.positive_prompt | self.llm
            positive_result = positive_chain.invoke({
                "question": question,
                "context": context
            })
            positive_opinion = positive_result.content
            
            # 부정 의견 추출
            negative_chain = self.negative_prompt | self.llm
            negative_result = negative_chain.invoke({
                "question": question,
                "context": context
            })
            negative_opinion = negative_result.content
            
            return {
                "positive_opinion": positive_opinion,
                "negative_opinion": negative_opinion
            }
        
        except Exception as e:
            print(f"감정 분석 에러: {str(e)}")
            return {
                "positive_opinion": "긍정 의견 분석 실패",
                "negative_opinion": "부정 의견 분석 실패"
            }
