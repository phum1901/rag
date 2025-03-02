from typing import Dict, List

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from src.configuration import Configuration
from src.prompt import (
    query_rewrite_system_prompt,
    query_rewrite_user_prompt,
)
from src.retrieval import get_retriever
from src.utils.model import load_chat_model
from src.utils.state import AgentState
from src.utils.tools import retrieve_tool


async def generate(state: AgentState, config: RunnableConfig):
    llm = load_chat_model(Configuration.from_runnable_config(config))
    ai_msg = await llm.ainvoke(state["messages"])
    return {"messages": ai_msg}


async def retrieve(state: AgentState, config: RunnableConfig) -> Dict[str, List[str]]:
    query = state["messages"][-1].content
    with get_retriever() as retriever:
        retrieved_documents = retriever.invoke(query, k=4)
    return {"query": query, "retrieved_documents": retrieved_documents}


async def rewrite_query(
    state: AgentState, config: RunnableConfig
) -> Dict[str, List[str]]:
    llm = load_chat_model(Configuration.from_runnable_config(config))
    query = state["messages"][-1].content
    prompt = ChatPromptTemplate(
        [("system", query_rewrite_system_prompt), ("human", query_rewrite_user_prompt)]
    )
    ai_msg = await (prompt | llm).ainvoke({"query": query})
    rewrited_query = ai_msg.content
    return {"query": query, "rewrited_query": rewrited_query}


def should_generate(state: AgentState):
    if state["retrieved_documents"]:
        return True
    else:
        return False


async def should_retrieve(state: AgentState, config: RunnableConfig):
    llm = load_chat_model(Configuration.from_runnable_config(config))
    llm_with_tools = llm.bind_tools([retrieve_tool])
    query = state["messages"][-1].content
    ai_msg = await llm_with_tools.ainvoke(state["messages"])
    return {"messages": [ai_msg], "query": query}
