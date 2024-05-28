import json
from typing import Any
from dataclasses import asdict
import numpy as np
from sefaria_llm_interface.common.topic import Topic
from sefaria_llm_interface.topic_prompt import TopicPromptSource

from experiments.topic_source_curation.cluster import Cluster, SummarizedSource

def _serialize_sources(sources: list[TopicPromptSource]) -> list[dict]:
    return [asdict(s) for s in sources]

def _deserialize_sources(raw_sources: list[dict]) -> list[TopicPromptSource]:
    return [TopicPromptSource(**s) for s in raw_sources]

def _serialize_clusters(clusters: list[Cluster]) -> list[dict]:
    serial = []
    for cluster in clusters:
        serialized_cluster = asdict(cluster)
        serialized_cluster['label'] = int(cluster.label)
        serialized_cluster['embeddings'] = [embedding.tolist() for embedding in cluster.embeddings]
        serialized_cluster['items'] = [source.serialize() for source in cluster.items]
        serial.append(serialized_cluster)
    return serial

def _deserialize_clusters(raw_clusters: list[dict]) -> list[Cluster]:
    return [_build_summarized_source_cluster(c) for c in raw_clusters]

def _build_summarized_source_cluster(raw_cluster: dict) -> 'Cluster':
    return Cluster(
        raw_cluster['label'],
        [np.array(embedding) for embedding in raw_cluster['embeddings']],
        [SummarizedSource(**s) for s in raw_cluster['items']],
        raw_cluster['summary']
    )

def _get_loader_and_saver(file_prefix, serializer, deserializer):
    def saver(data: Any, topic: Topic) -> None:
        with open(f"_cache/{file_prefix}_{topic.slug}.json", "w") as fout:
            json.dump(serializer(data), fout, indent=2, ensure_ascii=False)
        # return input so that pipeline can continue
        return data

    def loader(topic: Topic) -> Any:
        with open(f"_cache/{file_prefix}_{topic.slug}.json", "r") as fin:
            return deserializer(json.load(fin))

    return loader, saver

"""
Init loaders
"""
load_clusters, save_clusters = _get_loader_and_saver("clusters", _serialize_clusters, _deserialize_clusters)
load_sources, save_sources = _get_loader_and_saver("gathered_sources", _serialize_sources, _deserialize_sources)
