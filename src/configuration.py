from langchain_core.runnables import RunnableConfig, ensure_config
from pydantic import BaseModel, Field


class Configuration(BaseModel):
    model: str = Field(default="gemini-2.0-flash")
    model_provider: str = Field(default="google_genai")
    temperature: float = 0.5

    @classmethod
    def from_runnable_config(cls, config: RunnableConfig = None):
        config = ensure_config(config)
        configurable = config.get("configurable", {})
        print(cls.model_fields)
        _fields = {f for f in cls.model_fields.keys()}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})
