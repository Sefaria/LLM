import os
import argparse
from sefaria.utils.util import wrap_chars_with_overlaps
from sklearn.model_selection import train_test_split
import srsly
from util.general import load_mongo_docs

GPT_PROMPT_END_INDICATOR = "\n\n###\n\n"
GPT_COMPLETION_END_INDICATOR = "###"

SPAN_LABEL_TO_CHAR_WRAPPER = {
    "Person": ["{{", "}}"],
    "Group-of-People": ["{{", "}}"],
    "Name-of-God": ["{{", "}}"],
    "Citation": ["{{", "}}"],
}


def _get_wrap_chars(label):
    return SPAN_LABEL_TO_CHAR_WRAPPER.get(label, None)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('training_filename')
    parser.add_argument('validation_filename')
    parser.add_argument('-m', '--db-host', dest='db_host')
    parser.add_argument('-p', '--db-port', dest='db_port', type=int)
    parser.add_argument("-u", "--user", dest="user", default="", nargs="?")
    parser.add_argument("-r", "--replicaset", dest="replicaset", default="", nargs="?")
    return parser


def create_gpt_training_prompts(citation_docs):
    return [
        {"prompt": create_get_prompt_input(doc), "completion": create_gpt_prompt_output(doc)}
        for doc in citation_docs
    ]


def get_wrapped_citation(citation, label):
    wrapper_start, wrapper_end = _get_wrap_chars(label)
    return f"{wrapper_start}{citation}{wrapper_end}", len(wrapper_start), len(wrapper_end)


def create_get_prompt_input(citation_doc):
    return f"{citation_doc['text']} {GPT_PROMPT_END_INDICATOR}"


def create_gpt_prompt_output(citation_doc):
    chars_to_wrap = [(span['start'], span['end'], span['label']) for span in citation_doc['spans'] if span['label'] in SPAN_LABEL_TO_CHAR_WRAPPER]
    return f" {wrap_chars_with_overlaps(citation_doc['text'], chars_to_wrap, get_wrapped_citation)}{GPT_COMPLETION_END_INDICATOR}"


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    password = os.getenv('MONGO_PASSWORD')
    citation_docs = load_mongo_docs(args.input, args.db_host, args.db_port, args.user, password, args.replicaset)
    citation_docs = [doc for doc in citation_docs if doc['answer'] == 'accept']
    gpt_training = create_gpt_training_prompts(citation_docs)
    training_data, validation_data = train_test_split(gpt_training, random_state=613, train_size=0.8)
    print("TRAINING SIZE:", len(training_data))
    print("VALIDATION SIZE:", len(validation_data))
    srsly.write_jsonl(args.training_filename, training_data)
    srsly.write_jsonl(args.validation_filename, validation_data)
