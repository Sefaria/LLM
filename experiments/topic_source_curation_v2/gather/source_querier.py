from abc import ABC, abstractmethod
import django
django.setup()
from typing import Union
from sefaria.model.text import Ref
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source
from langchain_community.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from langchain_voyageai.embeddings import VoyageAIEmbeddings
from util.sefaria_specific import filter_invalid_refs


class SourceQuerierFactory:

    @staticmethod
    def create(typ) -> Union['AbstractSourceQuerier', 'SourceQuerierComposer']:
        if typ == "neo4j":
            return Neo4jSourceQuerier()
        if typ == "chroma_voyageai":
            return VoyageAIChromaSourceQuerier()
        if typ == "chroma_openai":
            return OpenAIChromaSourceQuerier()
        if typ == "chroma_openai_he":
            # currently produces very poor results
            return OpenAIHeChromaSourceQuerier()
        if typ == "chroma_all":
            return SourceQuerierComposer([
                SourceQuerierFactory.create('chroma_voyageai'),
                SourceQuerierFactory.create('chroma_openai')
            ])
        raise Exception("Type not found", typ)


class AbstractSourceQuerier(ABC):

    def __init__(self):
        self.vector_db = self._get_vector_db()

    @classmethod
    @abstractmethod
    def _get_vector_db(cls):
        pass

    def query(self, query: str, top_k: int, score_threshold: float) -> tuple[list[TopicPromptSource], list[float]]:
        retrieved_docs = self.vector_db.similarity_search_with_relevance_scores(
            query.lower(), top_k, score_threshold=score_threshold
        )
        retrieved_docs = filter_invalid_refs(retrieved_docs, key=lambda x: x[0].metadata['ref'])
        docs, scores = list(zip(*retrieved_docs)) if len(retrieved_docs) > 0 else ([], [])
        sources = [_make_topic_prompt_source(Ref(doc.metadata['ref']), '', with_commentary=False) for doc in docs]
        return sources, scores


class AbstractChromaSourceQuerier(AbstractSourceQuerier):
    persist_directory = None
    embedding_function = None

    @classmethod
    def _get_vector_db(cls):
        return Chroma(persist_directory=cls.persist_directory, embedding_function=cls.embedding_function)


class VoyageAIChromaSourceQuerier(AbstractChromaSourceQuerier):
    persist_directory = '../embedding/.chromadb'
    embedding_function = VoyageAIEmbeddings(model="voyage-large-2-instruct")


class OpenAIChromaSourceQuerier(AbstractChromaSourceQuerier):
    persist_directory = '../embedding/.chromadb_openai'
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")


class OpenAIHeChromaSourceQuerier(AbstractChromaSourceQuerier):
    persist_directory = '../embedding/.chromadb_openai_he'
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")


class Neo4jSourceQuerier(AbstractSourceQuerier):
    model_api = {
        'embedding_model': 'text-embedding-ada-002',
    }
    db = {
        'db_url': 'bolt://localhost:7689',
        'db_username': 'neo4j',
        'db_password': 'password',
    }

    @classmethod
    def _get_vector_db(cls):
        return Neo4jVector.from_existing_index(
            OpenAIEmbeddings(model=cls.model_api['embedding_model']),
            index_name="index",
            url=cls.db['db_url'],
            username=cls.db['db_username'],
            password=cls.db['db_password'],
        )


class SourceQuerierComposer:

    def __init__(self, queriers: list[AbstractSourceQuerier]):
        self.queriers = queriers

    def query(self, query: str, top_k: int, score_threshold: float) -> tuple[list[TopicPromptSource], list[float]]:
        sources = []
        scores = []
        for q in self.queriers:
            temp_sources, temp_scores = q.query(query, top_k=top_k, score_threshold=score_threshold)
            sources.extend(temp_sources)
            scores.extend(temp_scores)
        return sources, scores

