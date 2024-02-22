from typing import List, Dict, Optional, Union
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

    def __init__(self, ref: str, categories: List[str], book_description: Dict[str, str], book_title: Dict[str, str],
                 comp_date: str, author_name: str, context_hint: str, text: Dict[str, str],
                 commentary: List[Union[dict, TopicPromptCommentary]] = None, surrounding_text: Dict[str, str]=None):
        self.ref = ref
        self.categories = categories
        self.book_description = book_description
        self.book_title = book_title
        self.comp_date = comp_date
        self.author_name = author_name
        self.context_hint = context_hint
        self.text = text
        self.commentary = [
            c if isinstance(c, TopicPromptCommentary) else TopicPromptCommentary(**c)
            for c in (commentary or [])
        ]
        self.surrounding_text = surrounding_text
