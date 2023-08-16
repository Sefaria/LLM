import django
django.setup()
import asyncio
import re
import openai
import csv
import tiktoken
import os
from sefaria.model import *
import time
from get_normalizer import get_normalizer

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_completions(prompt, model="gpt-4", **completion_kwargs):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        **completion_kwargs,
    )
    return response.choices


async def get_topic_prompt_prompt(tref, topic_slug, lang):
    prompt = f"{_get_topic_prompt_intro(lang)}" \
             f"{_get_topic_prompt_examples(lang)}" \
             f"{await _get_topic_prompt_task(tref, topic_slug, lang)}"
    return prompt


def _get_topic_prompt_intro(lang):
    full_language = "English" if lang == "en" else "Hebrew"
    prompt = "# Task\n" \
             "## Goal: Write description of a Jewish text such that it persuades the reader to read the full source" \
             "## Input: Input has the following format:\n" \
                 "Topic: <topic>\n" \
                 "Source Text: <source text>\n" \
                 "Source Author: <author>\n" \
                 "Source Publication Year: <year>\n" \
                 "Source Book Description (optional): <book description>" \
                 "Commentary (optional): when it exists, use commentary to inform understanding of `Source Text`." \
             "## Output: Output should be in the following format:\n" \
                 "Description: <description> (no more than 50 words, should assume basic knowledge of Jewish terms, " \
                 f"written in {full_language})\n" \
                 "Description Title: <title>"
    return prompt


def _get_topic_slugs_with_prompts():
    with open("input/Topics that have been curated June 2023 (slugs on site) - Sheet1.csv", "r") as fin:
        cin = csv.reader(fin)
        for row in cin:
            slug = row[0]
            yield slug


def _get_query_for_ref_topic_link_with_prompt(slug, lang):
    return {"toTopic": slug, f"descriptions.{lang}": {"$exists": True}}


def _get_topic_prompt_examples(lang):
    prompt = "# Examples:"
    normalizer = get_normalizer()
    for islug, slug in enumerate(_get_topic_slugs_with_prompts()):
        topic = Topic.init(slug)
        tlink = RefTopicLink().load(_get_query_for_ref_topic_link_with_prompt(slug, lang))
        if tlink is None:
            print("no prompts for", slug)
            continue

        source_text = normalizer.normalize(Ref(tlink.ref).text(lang).ja().flatten_to_string())
        example = f"\n\n{islug+1}) Source Text: {source_text}\nTopic: {topic.get_primary_title(lang)}\n" \
                  f"Description: {tlink.descriptions[lang]['prompt']}\nTitle: {tlink.descriptions[lang]['title']}"
        prompt += example
        curr_num_tokens = count_tokens(prompt)
        if curr_num_tokens > 6000:
            break

    return prompt


async def _get_topic_prompt_task(tref, topic_slug, lang):
    topic = Topic.init(topic_slug)
    oref = Ref(tref)
    index = oref.index
    desc_attr = f"{lang}Desc"
    book_desc = getattr(index, desc_attr, "N/A")  # getattr(index, "enShortDesc", getattr(index, "enDesc", "N/A"))
    composition_time_period = index.composition_time_period()
    pub_year = composition_time_period.period_string(lang)
    author_name = Topic.init(index.authors[0]).get_primary_title(lang) if len(index.authors) > 0 else "N/A"
    source_text = get_ref_text_with_fallback(oref, lang)
    category = index.get_primary_category()
    prompt = f"\n\n-----\n\n" \
           f"# Input\n" \
           f"Topic: {topic.get_primary_title('en')}\n" \
           f"Source Text: {source_text}\n" \
           f"Source Author: {author_name}\n" \
           f"Source Publication Year: {pub_year}"
    if True:  # category not in {"Talmud", "Midrash", "Tanakh"}:
        prompt += f"\nSource Book Description: {book_desc}"
    if category in {"Tanakh"}:
        from summarize_commentary import summarize_commentary
        commentary_summary = await summarize_commentary(tref, topic_slug, company='anthropic')
        prompt += f"\nCommentary: {commentary_summary}"
    return prompt


def get_raw_ref_text(oref: Ref, lang: str) -> str:
    return oref.text(lang).ja().flatten_to_string()


def get_ref_text_with_fallback(oref: Ref, lang: str) -> str:
    raw_text = get_raw_ref_text(oref, lang)
    if len(raw_text) == 0:
        other_lang = "en" if lang == "he" else "he"
        raw_text = get_raw_ref_text(oref, other_lang)

    normalizer = get_normalizer()
    return normalizer.normalize(raw_text)


def count_tokens(prompt, model="gpt-4"):
    encoding = tiktoken.encoding_for_model(model)
    num_tokens = len(encoding.encode(prompt))
    return num_tokens


async def get_topic_prompts(tref, topic_slug, lang):
    prompt = await get_topic_prompt_prompt(tref, topic_slug, lang)
    return prompt, get_completions(prompt, temperature=0.5, n=3)


async def print_topic_prompt(tref, topic_slug, lang):
    gpt_prompt, topic_prompts = get_topic_prompts(tref, topic_slug, lang)
    print("Num Tokens", count_tokens(gpt_prompt))
    for idx, choice in enumerate(topic_prompts):
        print(idx, choice.message["content"])


async def create_evaluation_csv_of_topic_prompts(lang):
    with open("output/topic_prompt_evaluation.csv", "w") as fout:
        cout = csv.DictWriter(fout, ['Topic', 'Ref', 'GPT 1', 'GPT 2', 'GPT 3', 'Manual'])
        cout.writeheader()
        for slug in list(_get_topic_slugs_with_prompts()):
            print("Slug", slug)
            topic = Topic.init(slug)
            topic_links = RefTopicLinkSet(_get_query_for_ref_topic_link_with_prompt(slug, lang))
            if len(topic_links) <= 1:
                # first topic link is used in example passed in gpt prompt
                continue
            topic_links = topic_links.array()[1:3]
            for topic_link in topic_links:
                manual_prompt = topic_link.descriptions[lang]
                _, topic_prompts = await get_topic_prompts(topic_link.ref, slug, lang)
                row = {
                    "Topic": topic.get_primary_title('en'),
                    "Ref": topic_link.ref,
                    "Manual": f"Description: {manual_prompt['prompt']}\nDescription Title: {manual_prompt['title']}"
                }
                for idx, choice in enumerate(topic_prompts):
                    row[f"GPT {idx+1}"] = choice.message['content']
                cout.writerow(row)
                time.sleep(30)


if __name__ == '__main__':
    asyncio.run(create_evaluation_csv_of_topic_prompts("en"))

"""
Does badly on Leviticus 7:11-13 Gratitude
"""
