from typing import Union
from openai import OpenAI
from basic_langchain.cache import sqlite_cache
import numpy as np


class OpenAIEmbeddings:

    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        return np.array([d.embedding for d in self._call_embedding_api(documents).data])


    @sqlite_cache('embedding')
    def _call_embedding_api(self, documents: Union[list[str], str]):
        return self.client.embeddings.create(
            input=documents,
            model=self.model
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]
