import django
django.setup()
from sefaria.model.topic import Topic as SefariaTopic
from sefaria.model.text import Ref
from functools import partial
from basic_langchain.schema import SystemMessage, HumanMessage
from basic_langchain.chat_models import ChatOpenAI, ChatAnthropic
from util.general import get_by_xml_tag, run_parallel, summarize_text
from util.webpage import get_webpage_text
from util.sefaria_specific import filter_invalid_refs, convert_trefs_to_sources, remove_refs_from_same_category
from sefaria_llm_interface.common.topic import Topic
from sefaria.helper.topic import get_topic

def get_urls_for_topic_from_topic_object(topic: Topic) -> list[str]:
    sefaria_topic = SefariaTopic.init(topic.slug)
    assert isinstance(sefaria_topic, SefariaTopic)
    url_fields = [["enWikiLink", "heWikiLink"], ["enNliLink", "heNliLink"], ["jeLink"]]
    urls = []
    for fields_by_priority in url_fields:
        for field in fields_by_priority:
            value = sefaria_topic.get_property(field)
            if value is not None:
                urls.append(value)
                break
    return urls


def generate_topic_description(topic: Topic, text: str) -> str:
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system = SystemMessage(content="You are a Jewish teacher well versed in all Jewish texts and customs. Given text about a Jewish topic, wrapped in <text>, write a description for this topic for newcomers to Judaism. Topic is wrapped in <topic> tags. Output description should be no more than 2 sentences and wrapped in <description> tags.")
    human = HumanMessage(content=f"<topic>{topic.title['en']}</topic>\n<text>{text}</text>")
    response = llm([system, human])
    return get_by_xml_tag(response.content, "description")


def get_or_generate_topic_description(topic: Topic, verbose=True) -> str:
    # TODO create a general approach for deciding when a description isn't needed
    generate_desc = {'abraham-in-egypt'}
    description = topic.description.get('en', '')
    if topic.slug not in generate_desc:
        # these topics are better with just their titles. any description will limit their scope unnecessarily.
        return description
    if not description:
        description = get_topic_description_from_webpages(topic)
        if description and verbose:
            print('Generated desc from webpage:', description)
    if not description:
        description = get_topic_description_from_top_sources(topic, verbose=verbose)
        if description and verbose:
            print('Generated desc from sources:', description)
    return description


def get_topic_description_from_webpages(topic: Topic):
    urls = get_urls_for_topic_from_topic_object(topic)
    if len(urls) == 0:
        return
    text = get_webpage_text(urls[0])
    return generate_topic_description(topic, text)


def get_topic_description_from_top_sources(topic: Topic, verbose=True):
    top_trefs = get_top_trefs_from_slug(topic.slug, top_n=15)
    top_trefs = [r.normal() for r in remove_refs_from_same_category([Ref(tref) for tref in top_trefs], 2)][:6]
    top_sources = convert_trefs_to_sources(top_trefs)
    llm = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
    summaries = run_parallel([source.text['en'] for source in top_sources],
                             partial(summarize_text, llm=llm, max_words=30),
                             max_workers=2, desc="summarizing topic sources for description", disable=not verbose)
    bullet_point_str = f"{topic.title['en']} - Bullet Points:"
    for summary in summaries:
        bullet_point_str += f"- {summary.strip()}\n"
    return generate_topic_description(topic, bullet_point_str.strip())


def get_top_trefs_from_slug(slug, top_n=10) -> list[str]:
    out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
    try:
        trefs = [d['ref'] for d in out['refs']['about']['refs'] if not d['is_sheet']]
        trefs = filter_invalid_refs(trefs[:top_n])
        return trefs
    except KeyError:
        print('No refs found for {}'.format(slug))
        return []
