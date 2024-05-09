from abc import ABC, abstractmethod
from typing import Union
from openai import OpenAI
import voyageai
from basic_langchain.cache import sqlite_cache
import numpy as np


class AbstractEmbeddings(ABC):

    def embed_documents(self, documents: list[str]) -> np.ndarray:
        return np.array(self._call_embedding_api(documents))

    def embed_query(self, query: str) -> np.ndarray:
        return self.embed_documents([query])[0]

    @abstractmethod
    def _call_embedding_api(self, documents: Union[list[str], str]) -> list[list[float]]:
        pass


class OpenAIEmbeddings(AbstractEmbeddings):
    def __init__(self, model: str = "text-embedding-3-small"):
        self.client = OpenAI()
        self.model = model

    @sqlite_cache('embedding')
    def _call_embedding_api(self, documents: Union[list[str], str]):
        return [d.embedding for d in self.client.embeddings.create(
            input=documents,
            model=self.model
        ).data]


class VoyageAIEmbeddings(AbstractEmbeddings):

    def __init__(self, model: str):
        self.client = voyageai.Client()
        self.model = model

    @sqlite_cache('embedding')
    def _call_embedding_api(self, documents: Union[list[str], str]) -> list[list[float]]:
        if isinstance(documents, str):
            documents = [documents]
        return self.client.embed(
            documents,
            model=self.model
        ).embeddings

