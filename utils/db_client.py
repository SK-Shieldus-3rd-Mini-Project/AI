from langchain_community.vectorstores import Chroma
from utils.embedder import get_embeddings
from utils.config import settings

def get_vectorstore(collection_name="analyst_reports"):
    """ChromaDB 벡터스토어 반환"""
    embeddings = get_embeddings()
    vectorstore = Chroma(
        persist_directory=settings.chroma_db_path,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vectorstore
