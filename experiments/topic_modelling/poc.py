from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from collections import defaultdict


def query(q):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=".chromadb_openai", embedding_function=embeddings)
    ret = db.similarity_search_with_relevance_scores(q, k=100, score_threshold=0.4)
    for yo, score in ret:
        print("-----")
        print(score)
        print(yo.page_content)
        print(yo.metadata['Ref'])
    print("DB size", len(db.get()["ids"]))

def slugs_string_to_list(slugs_string):
    return [s for s in slugs_string.split('$') if s]

if __name__ == '__main__':
    # query('כורש מכונה "משיח" (המשוח) בספר ישעיהו (ישעיהו 45:1), מונח שבדרך כלל שמור למלכים ולכוהנים יהודים, מדגיש את תפקידו הייחודי בהיסטוריה היהודית.')
    query = """
In a measure David was indebted for his life to Adam. At first only three hours of existence had been allotted to him. When God caused all future generations to pass in review before Adam, he besought God to give David seventy of the thousand years destined for him. A deed of gift, signed by God and the angel Metatron, was drawn up. Seventy years were legally conveyed from Adam to David, and in accordance with Adam's wishes, beauty, dominion, and poetical gift went with them.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=".chromadb_openai", embedding_function=embeddings)
    docs = db.similarity_search_with_relevance_scores(
        query.lower(), 10, score_threshold=0.3
    )
    suggested_slugs = defaultdict(lambda: 0)
    for doc, score in docs:
        print("-----")
        print(score)
        print(doc.metadata['Ref'])
        slugs = slugs_string_to_list(doc.metadata['Slugs'])
        for slug in slugs:
            suggested_slugs[slug] += 1
    print(suggested_slugs)