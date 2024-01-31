from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TopicPromptCommentary:
    ref: str
    text: Dict[str, str]


@dataclass
class TopicPromptSource:
    ref: str
    categories: List[str]
    book_description: Dict[str, str]
    book_title: Dict[str, str]
    comp_date: str
    author_name: str
    context_hint: str
    text: Dict[str, str]
    commentary: Optional[List[TopicPromptCommentary]] = None  # list of commentary in a given language
    surrounding_text: Optional[Dict[str, str]] = None  # section for Tanakh and sugya for Talmud

    @staticmethod
    def deserialize(serial):
        if serial.get('commentary', []):
            commentary = [
                    TopicPromptCommentary(**comment)
                    for comment in serial.get('commentary', [])
                ]
        else:
            commentary = []
        return TopicPromptSource(**{
            **serial,
            "commentary": commentary
        })
