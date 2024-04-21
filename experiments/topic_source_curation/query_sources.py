from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.embeddings.openai import OpenAIEmbeddings
model_api = {
    'embedding_model': 'text-embedding-ada-002',
}
db = {
    'db_url': 'bolt://localhost:7689',
    'db_username': 'neo4j',
    'db_password': 'password',
}
neo4j_vector = Neo4jVector.from_existing_index(
            OpenAIEmbeddings(model=model_api['embedding_model']),
            index_name="index",
            url=db['db_url'],
            username=db['db_username'],
            password=db['db_password'],
        )
if __name__ == '__main__':
   query = 'Which famous Jewish scholars, rabbis and philosophers lived in or were associated with the Alexandrian Jewish community?'
   top_k = 10
   retrieved_docs = neo4j_vector.similarity_search_with_relevance_scores(
       query.lower(), top_k
   )
   for doc in retrieved_docs:
       print(doc[0].metadata['source'])
       print(doc[1])
