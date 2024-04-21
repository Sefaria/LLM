from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings

class SourceQuerier:
    model_api = {
        'embedding_model': 'text-embedding-ada-002',
    }
    db = {
        'db_url': 'bolt://localhost:7689',
        'db_username': 'neo4j',
        'db_password': 'password',
    }

    def __init__(self):
        self.neo4j_vector = self._get_neo4j_vector()

    @classmethod
    def _get_neo4j_vector(cls):
        return Neo4jVector.from_existing_index(
                    OpenAIEmbeddings(model=cls.model_api['embedding_model']),
                    index_name="index",
                    url=cls.db['db_url'],
                    username=cls.db['db_username'],
                    password=cls.db['db_password'],
                )

    def query_sources(self, query, top_k, score_threshold):
        retrieved_docs = self.neo4j_vector.similarity_search_with_relevance_scores(
            query.lower(), top_k, score_threshold=score_threshold
        )
        return retrieved_docs


if __name__ == '__main__':
   query = 'Why are dogs portrayed mostly negatively in the Bible?'
   top_k = 10000
   querier = SourceQuerier()
   docs = querier.query_sources(query, top_k, 0.9)
   for doc in docs:
       print(doc[0].metadata['source'])
       print(doc[1])
   print(len(docs))
