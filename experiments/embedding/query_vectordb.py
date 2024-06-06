from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma


def query(q):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=".chromadb_openai", embedding_function=embeddings)
    ret = db.similarity_search_with_relevance_scores(q, k=100, score_threshold=0.4)
    for yo, score in ret:
        print("-----")
        print(score)
        print(yo.page_content)
        print(yo.metadata['ref'])
    print("DB size", len(db.get()["ids"]))


if __name__ == '__main__':
    # query('כורש מכונה "משיח" (המשוח) בספר ישעיהו (ישעיהו 45:1), מונח שבדרך כלל שמור למלכים ולכוהנים יהודים, מדגיש את תפקידו הייחודי בהיסטוריה היהודית.')
    query = """
In the Midrash, "Listen, daughter, etc." "A palace is burning, etc." It is difficult to understand the verse "Listen, etc." and afterwards "Incline your ear." However, it is written, "To the land that I will show you," and why did He not show him immediately? The matter is that the Holy One, Blessed be He, derives more satisfaction from the desire and yearning that every Jew has to draw close to Him and attain the words of His Torah than from one's actual knowledge. For this desire itself merits him to draw close and attain. It is proven that this is more important before Him, Blessed be He. The order for a Jew is that through yearning to attain and hearken to His voice, may He be blessed, he merits to hear. Afterwards, he must internalize the attainment in the depths of his heart, which will cause him to pay more attention in depth, to increase his yearning to hear, and he will merit to hear more. And so it continues always. There is no rest, for "man is born to toil," and the righteous one's rest is in the World to Come. In this world, there is only the effort to progress from level to level. This is the meaning of "a palace is burning." It was a novelty in Abraham's eyes, for everything must be in its place of rest and root. The Holy One, Blessed be He, answered him that this is His will, may He be blessed, that in this world there should be only effort and no rest. This is the meaning of "that I will show you," for He knew that he would not attain perfection in this world, only to constantly yearn all the days of his life. This is the meaning of "Listen, etc." and "Incline your ear, etc." I heard from my grandfather, may his memory be for a blessing, that "a palace is burning" is from the language of "burning" (דלקת), for everything constantly yearns and pursues perfection, as mentioned above. 
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    db = Chroma(persist_directory=".chromadb_openai", embedding_function=embeddings)
    docs = db.similarity_search_with_relevance_scores(
        query.lower(), 1000, score_threshold=0.5
    )
    for doc, score in docs:
        print("-----")
        print(score)
        print(doc.metadata['ref'])

