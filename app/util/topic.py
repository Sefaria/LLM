import django
django.setup()
from sefaria.model.topic import Topic as SefariaTopic
from functools import partial
from basic_langchain.schema import SystemMessage, HumanMessage
from basic_langchain.chat_models import ChatOpenAI, ChatAnthropic
from util.general import get_by_xml_tag, run_parallel, summarize_text
from util.webpage import get_webpage_text
from util.sefaria_specific import filter_invalid_refs, convert_trefs_to_sources
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


def generate_topic_description(text):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system = SystemMessage(content="You are a Jewish teacher well versed in all Jewish texts and customs. Given text about a Jewish topic, wrapped in <text>, summarize the text in a small paragraph. Summary should be wrapped in <summary> tags.")
    human = HumanMessage(content=f"<text>{text}</text>")
    response = llm([system, human])
    return get_by_xml_tag(response.content, "summary")


def get_or_generate_topic_description(topic: Topic) -> str:
    description = topic.description.get('en', '')
    if not description:
        description = get_topic_description_from_webpages(topic)
    if not description:
        description = get_topic_description_from_top_sources(topic)
    return description


def get_topic_description_from_webpages(topic: Topic):
    urls = get_urls_for_topic_from_topic_object(topic)
    if len(urls) == 0:
        return
    text = get_webpage_text(urls[0])
    return generate_topic_description(text)


def get_topic_description_from_top_sources(topic: Topic, verbose=True):
    top_trefs = get_top_trefs_from_slug(topic.slug, top_n=5)
    top_sources = convert_trefs_to_sources(top_trefs)
    llm = ChatAnthropic(model='claude-3-haiku-20240307', temperature=0)
    summaries = run_parallel([source.text['en'] for source in top_sources],
                             partial(summarize_text, llm=llm, max_words=30),
                             max_workers=2, desc="summarizing topic sources for description", disable=not verbose)
    bullet_point_str = f"{topic.title['en']} - Bullet Points:"
    for summary in summaries:
        bullet_point_str += f"- {summary.strip()}\n"
    return generate_topic_description(bullet_point_str.strip())


def get_top_trefs_from_slug(slug, top_n=10) -> list[str]:
    out = get_topic(True, slug, with_refs=True, ref_link_type_filters=['about', 'popular-writing-of'])
    try:
        trefs = [d['ref'] for d in out['refs']['about']['refs'] if not d['is_sheet']]
        trefs = filter_invalid_refs(trefs[:top_n])
        return trefs
    except KeyError:
        print('No refs found for {}'.format(slug))
        return []
