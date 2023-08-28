from typing import List, Tuple
import django
django.setup()

import re
import requests
from sefaria.model import *


def _get_sheet_from_api(sheet_id: int) -> dict:
    response = requests.get(f"https://www.sefaria.org/api/sheets/{sheet_id}")
    return response.json()


def _get_sheet_orefs(sheet: dict) -> List[Ref]:
    return [Ref(tref) for tref in sheet['includedRefs']]


def _get_topic_from_name(topic_name: str) -> Topic:
    topic_set = TopicSet({"titles.text": topic_name})
    unique_topic = [topic for topic in topic_set if topic.get_primary_title("en") == topic_name and topic.numSources > 1]
    if len(unique_topic) != 1:
        if len(unique_topic) == 0 and len(topic_set) == 1:
            unique_topic = topic_set
        else:
            raise ValueError(f"Not only one topic with name {topic_name}. Count {len(unique_topic)}")
    return unique_topic[0]


def _get_topic_name_from_sheet(sheet: dict) -> str:
    topic_name_match = re.search(r":(.+)$", sheet['title'])
    if not topic_name_match:
        raise ValueError(f"Couldn't find topic name in {sheet['title']}")
    return topic_name_match.group(1).strip()


def _get_topic_from_sheet(sheet: dict) -> Topic:
    topic_name = _get_topic_name_from_sheet(sheet)
    return _get_topic_from_name(topic_name)


def get_topic_and_orefs(sheet_id: int) -> Tuple[Topic, List[Ref]]:
    sheet = _get_sheet_from_api(sheet_id)
    topic = _get_topic_from_sheet(sheet)
    orefs = _get_sheet_orefs(sheet)
    return topic, orefs
