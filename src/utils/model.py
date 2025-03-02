from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel, Field

from src.configuration import Configuration


def load_chat_model(config: Configuration) -> BaseChatModel:
    return init_chat_model(
        model=config.model,
        model_provider=config.model_provider,
    )


class GraderOutput(BaseModel):
    """boolean for relevance check on retrieved documents."""

    is_relevant: bool = Field(
        description="True if document is relevant to the question otherwise False"
    )
    reason: str = Field(
        description="Reason why the document is relevant to the question."
    )
