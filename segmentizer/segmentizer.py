import django
django.setup()
from sefaria.model import *
from util.anthropic import get_completion_anthropic


def get_task_description(task="sentencizer"):
    if task == "sentencizer":
        return "Break up input into sentences\n"
    elif task == "segmentizer":
        return "Break up input into thematic sections. Each thematic section should discuss one central theme.\n"


def get_prompt(text_to_segmentize, task="sentencizer"):
    section_name = 'sentence' if task == 'sentencizer' else 'section'
    return "# Task\n" + get_task_description(task) + "## Input\n" \
           "Rabbinical Jewish Text. Input starts after \"----\"\n" \
           "## Output\n" \
           "Don't output any text besides for the input text and the verbatim text '$$$$$'" \
           f"Each {section_name} should be on its own line.\n" \
           f"After each {section_name}, output the verbatim text '$$$$$' on its own line." \
           f"----\n{text_to_segmentize}"


def segmentize(text_to_segmentize, task="sentencizer"):
    prompt = get_prompt(text_to_segmentize, task)
    completion = get_completion_anthropic(prompt)
    sentences = [sent.strip() for sent in completion.split('$$$$$') if len(sent.strip()) > 0]
    return sentences


def print_completion(sentences):
    print("====OUTPUT====")
    for sent in sentences:
        print(sent)
        print('-----')


if __name__ == '__main__':
    sents = segmentize("sentencizer")
    print_completion(sents)

