from langsmith import evaluate, Client
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_openai import ChatOpenAI
from langsmith.schemas import Example, Run


set_llm_cache(SQLiteCache(database_path=".langchain.db"))

def get_example(x: dict) -> dict:
    # llm = ChatOpenAI()
    # answer = llm.invoke(x['q']).content
    print(x)
    # return {'a': answer}
def check_answer(root_run: Run, example: Example) -> dict:
    score = 0
    if  example.outputs['a'] in root_run.outputs['a']:
        score = 1
    return {"key": "correct_answer", "score": score}

if __name__ == '__main__':
    client = Client()

    evaluate(
        get_example,
        data="topic-tagging-gold",
        evaluators=[check_answer],
        experiment_prefix="dummy_experiment",
    )