from dataclasses import dataclass
from typing import List, Dict, Union


@dataclass
class SheetScoringInput:
    # str version of id 
    sheet_id: str
    title: str
    sources: List[Dict[str, Union[str, Dict[str, str]]]]
    expanded_refs: str

