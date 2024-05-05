from abc import ABC, abstractmethod
import django
django.setup()
from sefaria.model.text import Ref
from sefaria.helper.llm.topic_prompt import _make_topic_prompt_source
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from sefaria_llm_interface.topic_prompt import TopicPromptSource
from langchain_voyageai.embeddings import VoyageAIEmbeddings


class SourceQuerierFactory:

    @staticmethod
    def create(typ) -> 'AbstractSourceQuerier':
        if typ == "neo4j":
            return Neo4jSourceQuerier()
        if typ == "chroma":
            return ChromaSourceQuerier()
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
            query.lower(), top_k
        )
        docs, scores = list(zip(*retrieved_docs)) if len(retrieved_docs) > 0 else ([], [])

        sources = [_make_topic_prompt_source(Ref(doc.metadata['ref']), '', with_commentary=False) for doc in docs]
        return sources, scores



class ChromaSourceQuerier(AbstractSourceQuerier):
    persist_directory = '../embedding/.chromadb'

    @classmethod
    def _get_vector_db(cls):
        return Chroma(persist_directory=cls.persist_directory, embedding_function=VoyageAIEmbeddings(model="voyage-large-2-instruct"))


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
