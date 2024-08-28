import json
from util.general import run_parallel
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document




if __name__ == '__main__':
    with open("topic_modelling_training_set.json", 'r') as file:
        items = json.load(file)
    formatted_items = []
    for item in items:
        formatted = {
            'text': item['English'],
            'metadata': {key: value for key, value in item.items() if key!='English'}
                     }
        formatted_items.append(formatted)

    docs = []
    for item in formatted_items:
        docs.append(Document(page_content=item['text'], metadata=item['metadata']))
    print(docs)