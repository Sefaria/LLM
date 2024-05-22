import requests
from readability import Document

def get_webpage_text(url: str) -> str:
    response = requests.get(url)
    doc = Document(response.content)
    return f"{doc.title()}\n{doc.summary()}"
