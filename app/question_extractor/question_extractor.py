import django
django.setup()
from app.util.openai import get_completion_openai, count_tokens_openai
from sefaria.model import *


def _get_prompt_task():
    return "# Background\n" \
           "You are a Jewish scholar knowledge in all Torah and Jewish texts.\n" \
           "# Task\n" \
           "Given input in the following format:\n" \
           "## Base Text: A verse from the Torah\n" \
           "## Commentary Text: A medieval Jewish commentator's comment of the base text\n" \
           "1) Output a summary of the commentary followed by the string '###'\n" \
           "2) Output a list of questions that a reader might have while reading the base text that the commentary " \
           "answers.\n" \
           "3) Output the corresponding answer for each question.\n" \
           "# Rules\n" \
           "Each question on its own line in a numbered list.\n" \
           "Each answer should be on its own line in a numbered list.\n" \
           "Each question/answer pair should share the same number.\n" \
           "Each answer should be a verbatim quote from the input.\n" \
           "Questions should be written assuming a knowledgeable Jewish audience\n"


def _get_tref_text(tref):
    return Ref(tref).text("en").ja().flatten_to_string()


def _get_prompt_input(base_tref, commentary_tref):
    return f"#Input\n" \
           f"## Base Text\n" \
           f"{_get_tref_text(base_tref)}\n" \
           f"## Commentary Text\n" \
           f"{_get_tref_text(commentary_tref)}\n"


def get_prompt(base_tref, commentary_tref):
    return f"{_get_prompt_task()}{_get_prompt_input(base_tref, commentary_tref)}"


def extract_questions(base_tref, commentary_tref):
    prompt = get_prompt(base_tref, commentary_tref)
    print(f"Prompt length: {count_tokens_openai(prompt)}")
    completion = get_completion_openai(prompt)
    return completion


if __name__ == '__main__':
    """
    """
    print(extract_questions("Deuteronomy 3:23", "Rashi on Deuteronomy 3:23:1"))

