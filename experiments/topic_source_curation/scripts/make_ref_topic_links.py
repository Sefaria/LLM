import random
import json
from experiments.topic_source_curation.curator import get_topics_to_curate
import django

from sefaria.model import RefTopicLink
from sefaria.model import Topic as SefariaTopic
from sefaria.helper.llm.topic_prompt import make_llm_topic
django.setup()
random.seed(45612)


def _make_ref_topic_link(topic, tref, context, i):
    return {
        "toTopic": topic.slug,
        "ref": tref,
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
                "ai_context": context,
                "published": False,
                "review_state": "not reviewed",
            }
        }
    }


def save_ref_topic_links():
    import json
    with open("data/private/ref_topic_links.json", "r") as fin:
        links = json.load(fin)
    len(links)
    for link in links:
        RefTopicLink(link).save()

def _generate_all_prompts():
    from tqdm import tqdm
    slugs_to_generate = {l.toTopic for l in RefTopicLinkSet({"generatedBy": "auto-curator"})}
    slugs_to_generate = [
        'balaam',
        'caleb',
        'parents',
        'parah-adumah',
        'hunger',
        'disability',
        'aarons-death',
        'josephs-dream',
        'empathy',
        'leviathan',
        'memory',
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
    # topics = random.sample(get_topics_to_curate(), 50)
    topics = [make_llm_topic(SefariaTopic.init(slug)) for slug in [
        'creation-of-man'
    ]]
    for topic in topics:
        print(topic.slug)
        with open(f"output/curation_{topic.slug}.json", "r") as fin:
            curation = json.load(fin)
            for i, entry in enumerate(curation):
                links += [_make_ref_topic_link(topic, entry['ref'], entry['context'], len(curation)-i)]
    with open(f"output/ref_topic_links.json", "w") as fout:
        json.dump(links, fout, ensure_ascii=False, indent=2)
