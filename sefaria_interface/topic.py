from dataclasses import dataclass
from typing import Dict


@dataclass
class Topic:
    slug: str
    description: Dict[str, str]  # keyed by lang
    title: Dict[str, str]  # keyed by lang
