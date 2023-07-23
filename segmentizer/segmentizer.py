import django
django.setup()
from sefaria.model import *
from llm.openai.openai_util.util import get_completion_openai, count_tokens_openai


def get_task_description(task="sentencizer"):
    if task == "sentencizer":
        return "Break up input into sentences\n"
    elif task == "segmentizer":
        return "Break up input into thematic sections. Each thematic section should discuss one central theme.\n"


def get_prompt(task="sentencizer"):
    return "# Task\n" + get_task_description(task) + "## Input\n" \
           "Rabbinical Jewish Text. Input starts after \"----\"\n" \
           "## Output\n" \
           f"Input text with newlines to break up each {'sentence' if task == 'sentencizer' else 'section'}\n" \
           "----\n"


if __name__ == '__main__':
    ref = Ref("Mishnah Kilayim 4:1")
    prompt = get_prompt("segmentizer")
    text = ref.text('en').text
    prompt += text
    print(count_tokens_openai(prompt))
    print(get_completion_openai(prompt))
