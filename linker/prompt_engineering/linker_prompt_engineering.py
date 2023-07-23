from util.general import load_mongo_docs
from util.openai import count_tokens_openai, get_completion_openai
from sefaria.utils.util import wrap_chars_with_overlaps


def get_prompt(input_text):
    return f"{get_task_description()}{get_examples_string()}{get_input_string(input_text)}"


def get_input_string(input_text):
    return f"# Input\n{input_text}"


def get_task_description():
    return "# Task:\n" \
           "Annotate every citation to a Jewish source in input\n" \
           "Each citation should be wrapped in double brackets\n"


def get_examples_string():
    citation_docs = list(load_mongo_docs("webpages_en_output3"))
    citation_docs.sort(key=lambda x: len(x["spans"]), reverse=True)
    examples_string = "# Examples\n## Format\nEach example appears on its own line\nEach citation is wrapped in double " \
                      "brackets\n## Examples\n"
    for idoc, doc in enumerate(citation_docs):
        if count_tokens_openai(examples_string) > 6000:
            print(idoc)
            break
        examples_string += f"{wrap_citations_in_doc(doc)}\n"
    print(count_tokens_openai(examples_string))
    return examples_string


def get_wrapped_citation(citation, metadata):
    return f"[[{citation}]]", 2, 2


def wrap_citations_in_doc(citation_doc):
    chars_to_wrap = [(span['start'], span['end'], None) for span in citation_doc['spans']]
    return wrap_chars_with_overlaps(citation_doc['text'], chars_to_wrap, get_wrapped_citation)


def get_input_text():
    return "\n".join(open("input/english_citation_input.txt").readlines())


if __name__ == '__main__':
    prompt = get_prompt(get_input_text())
    print(prompt)
    print(count_tokens_openai(prompt))
    print(get_completion_openai(prompt))
