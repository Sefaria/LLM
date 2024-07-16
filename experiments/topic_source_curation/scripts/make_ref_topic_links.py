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


def find_topics_that_didnt_generate():
    from tqdm import tqdm
    slugs_to_generate = [
        "lilies",
        "arbeh",
        "man-and-animals",
        "mice",
        "mountains",
        "mules",
        "myrtles",
        "night",
        "oxen",
        "pigs",
        "rams",
        "ravens",
        "rivers",
        "roosters",
        "science",
        "scorpions",
        "seas1",
        "sherets",
        "snakes1",
        "snow",
        "spring",
        "springs",
        "sun",
        "sustainability",
        "the-elements",
        "the-moon",
        "the-ocean",
        "vines",
        "water",
        "weather",
        "wild-animals",
        "willows",
        "winds",
        "wolves",
        "worms",
        "yavneh",
        "edom",
        "the-four-kingdoms",
        "alexandria",
        "the-four-rivers",
        "distinctions-of-the-land-of-israel",
        "babel",
        "beersheba",
        "beit-el",
        "beit-hamikdash",
        "galil",
        "germany",
        "damascus",
        "har-gerizim-vehar-eival",
        "moriah",
        "outside-of-israel",
        "the-land-of-judah",
        "greece",
        "jerusalem",
        "kush",
        "levite-cities",
        "midian",
        "moab",
        "partitions",
        "encampment-of-israel",
        "cave-of-the-patriarchs",
        "tabernacle-courtyard",
        "the-nile-river",
        "nineveh",
        "sodom",
        "syria",
        "ever-hayarden",
        "cities",
        "ammon",
        "gemorrah",
        "tyre",
        "zion",
        "rome",
        "the-tabernacle-in-shiloh",
        "shechem",
        "heavens",
        "sharon",
        "tarshish",
        "beit-midrash",
        "temple-mount",
        "second-temple",
        "en-dor",
        "jericho",
        "israel",
        "ashur",
        "food",
        "honey",
        "olives",
        "milk",
        "wine",
        "fresh-grain",
        "maror",
        "planting",
        "peppers",
        "fruit",
        "flour",
        "drinking",
    ]

    bad_slugs = []
    for slug in tqdm(slugs_to_generate):
        topic_links = RefTopicLinkSet({"descriptions.en.ai_title": {"$exists": True}, "toTopic": slug})
        if len(topic_links) == 0:
            bad_slugs.append(slug)
    for slug in tqdm(bad_slugs):
        print(slug)
        _generate_prompts_for_slug(slug)

def save_ref_topic_links():
    import json
    from tqdm import tqdm
    with open("scripts/ref_topic_links.json", "r") as fin:
        links = json.load(fin)
    len(links)
    for link in tqdm(links):
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
        "lilies",
        "arbeh",
        "man-and-animals",
        "mice",
        "mountains",
        "mules",
        "myrtles",
        "night",
        "oxen",
        "pigs",
        "rams",
        "ravens",
        "rivers",
        "roosters",
        "science",
        "scorpions",
        "seas1",
        "sherets",
        "snakes1",
        "snow",
        "spring",
        "springs",
        "sun",
        "sustainability",
        "the-elements",
        "the-moon",
        "the-ocean",
        "vines",
        "water",
        "weather",
        "wild-animals",
        "willows",
        "winds",
        "wolves",
        "worms",
        "yavneh",
        "edom",
        "the-four-kingdoms",
        "alexandria",
        "the-four-rivers",
        "distinctions-of-the-land-of-israel",
        "babel",
        "beersheba",
        "beit-el",
        "beit-hamikdash",
        "galil",
        "germany",
        "damascus",
        "har-gerizim-vehar-eival",
        "moriah",
        "outside-of-israel",
        "the-land-of-judah",
        "greece",
        "jerusalem",
        "kush",
        "levite-cities",
        "midian",
        "moab",
        "partitions",
        "encampment-of-israel",
        "cave-of-the-patriarchs",
        "tabernacle-courtyard",
        "the-nile-river",
        "nineveh",
        "sodom",
        "syria",
        "ever-hayarden",
        "cities",
        "ammon",
        "gemorrah",
        "tyre",
        "zion",
        "rome",
        "the-tabernacle-in-shiloh",
        "shechem",
        "heavens",
        "sharon",
        "tarshish",
        "beit-midrash",
        "temple-mount",
        "second-temple",
        "en-dor",
        "jericho",
        "israel",
        "ashur",
        "food",
        "honey",
        "olives",
        "milk",
        "wine",
        "fresh-grain",
        "maror",
        "planting",
        "peppers",
        "fruit",
        "flour",
        "drinking",
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
    print(len(get_topics_to_curate()))
    all_topics = get_topics_to_curate()
    istart = next((i for i, topic in enumerate(all_topics) if topic.slug == "lilies"), None)
    iend = next((i for i, topic in enumerate(all_topics) if topic.slug == "drinking"), None)
    for topic in all_topics[istart:iend+1]:
        print(f'"{topic.slug}",')
        try:
            with open(f"output/curation_{topic.slug}.json", "r") as fin:
                curation = json.load(fin)
                for i, entry in enumerate(curation):
                    links += [_make_ref_topic_link(topic, entry['ref'], entry['context'], len(curation)-i)]
        except:
            print("SKIP", topic.slug)
    with open(f"output/ref_topic_links.json", "w") as fout:
        json.dump(links, fout, ensure_ascii=False, indent=2)
