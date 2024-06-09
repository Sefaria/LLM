"""
Looking to compare various models for filtering out topics that aren't relevant to the topic after the gather stage
"""
import json
from sefaria_llm_interface.topic_prompt import TopicPromptSource
GATHER_SOURCES_FILE_BASE = "../_cache/gathered_sources"

def compare_trefs(a, b):
    count_b_not_in_a = 0
    count_a_not_in_b = 0
    for bb in b:
        if bb not in a:
            print(bb)
            count_b_not_in_a += 1
    print('-----')
    for aa in a:
        if aa not in b:
            print(aa)
            count_a_not_in_b += 1
    print("B not in A", count_b_not_in_a)
    print("A not in B", count_a_not_in_b)

def load_trefs_from_json(slug, model) -> list[str]:
    with open(f"{GATHER_SOURCES_FILE_BASE}_{slug}_{model}.json") as fin:
        raw_sources = json.load(fin)
        return [TopicPromptSource(**s).ref for s in raw_sources]

def compare_models(slug, model1, model2):
    trefs1 = load_trefs_from_json(slug, model1)
    trefs2 = load_trefs_from_json(slug, model2)
    for tref in trefs2:
        print(tref)
    # compare_trefs(trefs1, trefs2)

if __name__ == '__main__':
    compare_models('ulla', 'gpt4', 'gpt4o')


