from typing import Union
import numpy as np
from dataclasses import dataclass, asdict
from util.cluster import AbstractClusterItem
from sefaria_llm_interface.topic_prompt import TopicPromptSource


@dataclass
class SummarizedSource(AbstractClusterItem):
    source: TopicPromptSource
    summary: str

    def __init__(self, source: Union[TopicPromptSource, dict], summary: str, embedding: np.ndarray = None):
        self.source = source if isinstance(source, TopicPromptSource) else TopicPromptSource(**source)
        self.summary = summary
        self.embedding = np.array(embedding) if embedding is not None else None

    def serialize(self) -> dict:
        serial = asdict(self)
        serial['embedding'] = self.embedding.tolist() if self.embedding is not None else None
        return serial

    def get_str_to_summarize(self) -> str:
        return self.summary or 'N/A'

    def get_str_to_embed(self) -> str:
        return self.summary
