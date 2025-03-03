import os
from contextlib import contextmanager
from typing import Dict

from langchain_core.embeddings import Embeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings


def get_embedding_model(
    model_name: str,
    model_kwargs: Dict = {},
    encode_kwargs: Dict = {},
) -> Embeddings:
    return HuggingFaceEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )


@contextmanager
def get_qdrant_retriever(embedding_model: Embeddings, *args, **kwargs):
    from langchain_qdrant import QdrantVectorStore
    from qdrant_client import QdrantClient
    from qdrant_client.http.models import Distance, VectorParams

    collection_name = os.getenv("COLLECTION_NAME")
    size = os.getenv("EMBEDDING_DIM")
    client = QdrantClient()  # TODO config

    if not client.collection_exists(collection_name):
        vectors_config = VectorParams(size=size, distance=Distance.COSINE)
        client.create_collection(collection_name, vectors_config=vectors_config)

    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name, embedding_model
    )
    yield vector_store.as_retriever()


@contextmanager
def get_retriever(*args, **kwargs):
    model_name = os.getenv("EMBEDDING_MODEL_NAME")
    embedding_model = get_embedding_model(model_name=model_name)
    with get_qdrant_retriever(embedding_model, *args, **kwargs) as retriever:
        yield retriever
