from typing import Annotated, List, Sequence, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class InputState(TypedDict):
    messages: Annotated[Sequence[AnyMessage], add_messages]


class AgentState(InputState):
    documents: List[Document]
    query: str
    rewrited_query: str


class IndexState:
    documents: Sequence[Document]


class Config:
    temperature: float = 0.5
