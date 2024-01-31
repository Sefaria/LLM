import dataclasses
import re
import django
django.setup()
from sefaria.model import *
from sefaria.client.wrapper import get_links
from sefaria.datatype.jagged_array import JaggedTextArray
from sefaria_interface.topic import Topic as TPTopic
from sefaria_interface.topic_prompt_source import TopicPromptSource, TopicPromptCommentary
from sefaria_interface.topic_prompt_input import TopicPromptInput
from util.sefaria_specific import get_ref_text_with_fallback


def get_commentary_for_tref(tref):
    library.rebuild_toc()
    commentary = []

    for link_dict in get_links(tref, with_text=True):
        if link_dict['category'] not in {'Commentary'}:
            continue
        if not link_dict['sourceHasEn']:
            continue
        temp_commentary = TopicPromptCommentary(link_dict['sourceRef'], {
            'en': JaggedTextArray(link_dict['text']).flatten_to_string(),
            'he': JaggedTextArray(link_dict['he']).flatten_to_string()
        })
        for lang in ('en', 'he'):
            temp_commentary.text[lang] = re.sub(r"<[^>]+>", " ", TextChunk.strip_itags(temp_commentary.text[lang]))
        commentary += [temp_commentary]
    return commentary


def _get_context_ref(segment_oref: Ref):
    if segment_oref.primary_category == "Tanakh":
        return segment_oref.section_ref()
    elif segment_oref.index.get_primary_corpus() == "Bavli":
        passage = Passage.containing_segment(segment_oref)
        return passage.ref()
    return None


def get_surrounding_text(oref: Ref):
    context_ref = _get_context_ref(oref)
    if context_ref:
        return {lang: get_ref_text_with_fallback(context_ref, 'en') for lang in ('en', 'he')}


def convert_sheet_to_topic_prompt_input(sefaria_topic, orefs, contexts):

    tp_topic = TPTopic(
        sefaria_topic.slug,
        getattr(sefaria_topic, 'description', {}),
        {
            'en': sefaria_topic.get_primary_title('en'),
            'he': sefaria_topic.get_primary_title('he')
        }
    )
    sources = []
    for oref, context in zip(orefs, contexts):
        assert isinstance(oref, Ref)
        index = oref.index
        text = {lang: get_ref_text_with_fallback(oref, 'en') for lang in ('en', 'he')}
        book_description = {lang: getattr(index, f"{lang}Desc", "N/A") for lang in ('en', 'he')}
        book_title = {lang: index.get_title(lang) for lang in ('en', 'he')}
        composition_time_period = index.composition_time_period()
        pub_year = composition_time_period.period_string("en") if composition_time_period else "N/A"
        try:
            author_name = Topic.init(index.authors[0]).get_primary_title("en") if len(index.authors) > 0 else "N/A"
        except AttributeError:
            author_name = "N/A"

        commentary = None
        if index.get_primary_category() == "Tanakh":
            commentary = get_commentary_for_tref(oref.normal())
        surrounding_text = get_surrounding_text(oref)
        sources += [
            TopicPromptSource(
                oref.normal(),
                index.categories,
                book_description,
                book_title,
                pub_year,
                author_name,
                context,
                text,
                commentary,
                surrounding_text
            )
        ]
    tp_input = TopicPromptInput("en", tp_topic, sources)
    return dataclasses.asdict(tp_input)

