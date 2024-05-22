import django
django.setup()
from sefaria.model.topic import Topic as SefariaTopic
from basic_langchain.schema import SystemMessage, HumanMessage
from basic_langchain.chat_models import ChatOpenAI
from util.general import get_by_xml_tag
from util.webpage import get_webpage_text
from sefaria_llm_interface.common.topic import Topic

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


def generate_topic_description(webpage_text):
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    system = SystemMessage(content="You are a Jewish teacher well versed in all Jewish texts and customs. Given text about a Jewish topic, wrapped in <text>, summarize the text in a small paragraph. Summary should be wrapped in <summary> tags.")
    human = HumanMessage(content=f"<text>{webpage_text}</text>")
    response = llm([system, human])
    return get_by_xml_tag(response.content, "summary")


def get_topic_description(topic: Topic):
    urls = get_urls_for_topic_from_topic_object(topic)
    if len(urls) == 0:
        return
    text = get_webpage_text(urls[0])
    return generate_topic_description(text)
