import random
import json
from experiments.topic_source_curation.curator import get_topics_to_curate
import django

from sefaria.model import RefTopicLink, Ref
from sefaria.model import Topic as SefariaTopic
from sefaria.helper.llm.topic_prompt import make_llm_topic
django.setup()
random.seed(45612)


def _make_ref_topic_link(topic, tref, context, i):
    return {
        "toTopic": topic.slug,
        "ref": Ref(tref).normal(),
        "linkType": "about",
        "class": "refTopic",
        "dataSource": "learning-team",
        "generatedBy": "auto-curator",
        "order": {
            "curatedPrimacy" : {
                "en" : i,
            },
        },
        "descriptions": {
            "en": {
                "ai_context": "N/A",
                "published": False,
                "review_state": "not reviewed",
            }
        }
    }


def save_ref_topic_links():
    import json
    with open("scripts/ref_topic_links.json", "r") as fin:
        links = json.load(fin)
    len(links)
    for link in links:
        try:
            existing_links = RefTopicLinkSet({
                "toTopic": link["toTopic"],
                "ref": link["ref"],
                "linkType": link["linkType"],
                "dataSource": link["dataSource"],
            })
        except Exception as e:
            continue
            print(e)
        if existing_links:
            existing_links.delete()
        RefTopicLink(link).save()

def _generate_all_prompts():
    from tqdm import tqdm
    slugs_to_generate = [
        "naaman",
        "nephilim",
        "naftali",
        "sisera",
        "sennacherib",
        "serah-the-daughter-of-asher",
        "iddo",
        "obadiah",
        "og",
        "uzziah",
        "ezra",
        "achan",
        "the-sons-of-eli",
        "eli",
        "amos",
        "amram",
        "amasa",
        "efron",
        "er-(firstborn-son-of-judah)",
        "esau",
        "potiphar",
        "the-concubine-of-givah",
        "pharaoh",
        "zelophehad",
        "keturah",
        "cain",
    ]

    for slug in tqdm(slugs_to_generate):
        _generate_prompts_for_slug(slug)

def _generate_prompts_for_slug(slug):
    from sefaria.helper.llm.tasks import generate_and_save_topic_prompts
    from sefaria.helper.llm.topic_prompt import get_ref_context_hints_by_lang
    ref_topic_links = [l.contents() for l in RefTopicLinkSet({"toTopic": slug, "generatedBy": "auto-curator"})]
    topic = Topic.init(slug)
    for lang, ref__context_hints in get_ref_context_hints_by_lang(ref_topic_links).items():
        orefs, context_hints = zip(*ref__context_hints)
        generate_and_save_topic_prompts(lang, topic, orefs, context_hints)


if __name__ == '__main__':
    links = []
    # topics = get_topics_to_curate()[94:120]
    topics = [make_llm_topic(SefariaTopic.init('abel'))]
    for topic in topics:
        print(f'"{topic.slug}",')
        with open(f"output/curation_{topic.slug}.json", "r") as fin:
            curation = json.load(fin)
            for i, entry in enumerate(curation):
                links += [_make_ref_topic_link(topic, entry['ref'], entry['context'], len(curation)-i)]
    with open(f"output/ref_topic_links.json", "w") as fout:
        json.dump(links, fout, ensure_ascii=False, indent=2)
