from typing import List

from langchain_core.tools import tool

from src.retrieval import get_retriever


@tool
def retrieve_tool(query: str) -> List[str]:
    """retrieve documents that is relevant to question"""
    with get_retriever() as retriever:
        documents = retriever.invoke(query, k=2)
        # return [doc.page_content for doc in documents]
        return documents
