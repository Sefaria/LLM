from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class TopicPromptSource:
    ref: str
    categories: List[str]
    book_description: str
    comp_date: str
    author_name: str
    context_hint: str
    text: Dict[str, str]
    commentary: Optional[List[str]] = None  # list of commentary in a given language
    surrounding_text: Optional[Dict[str, str]] = None  # section for Tanakh and sugya for Talmud
