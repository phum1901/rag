from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from src.configuration import Configuration
from src.utils.nodes import generate, retrieve_tool, should_retrieve
from src.utils.state import AgentState, InputState

retrieval_builder = StateGraph(
    state_schema=AgentState, input=InputState, config_schema=Configuration
)

retrieve = ToolNode([retrieve_tool], messages_key="messages")

retrieval_builder.add_node("retrieve", retrieve)
retrieval_builder.add_node("should_retrieve", should_retrieve)
retrieval_builder.add_node("generate", generate)

retrieval_builder.add_edge("__start__", "should_retrieve")
retrieval_builder.add_conditional_edges(
    source="should_retrieve",
    path=tools_condition,
    path_map={"tools": "retrieve", "__end__": "__end__"},
)
retrieval_builder.add_edge("retrieve", "generate")
retrieval_builder.add_edge("generate", "__end__")


retrieval_graph = retrieval_builder.compile()
