import os
import argparse
from sefaria.utils.util import wrap_chars_with_overlaps
from sklearn.model_selection import train_test_split
import srsly
from util.general import load_mongo_docs
from constants import GPT_PROMPT_END_INDICATOR, GPT_COMPLETION_END_INDICATOR


SPAN_LABEL_TO_CHAR_WRAPPER = {
    "Person": ["{{", "}}"],
    "Group-of-People": ["{{", "}}"],
    "Name-of-God": ["{{", "}}"],
    "Citation": ["{{", "}}"],
}

SPAN_LABEL_TO_CLASSICATION_TAG = {
    "Person": "Person",
    "Group-of-People": "Group",
    "Name-of-God": "Person",
    "Citation": "Citation",
}


def _get_wrap_chars(label):
    return SPAN_LABEL_TO_CHAR_WRAPPER.get(label, None)


def _get_wrapped_citation(citation, label):
    wrapper_start, wrapper_end = _get_wrap_chars(label)
    return f"{wrapper_start}{citation}{wrapper_end}", len(wrapper_start), len(wrapper_end)


def get_window_around_match(start_char:int, end_char:int, text:str, window:int=10) -> tuple:
    before_text = text[:start_char]
    before_window_words = list(filter(lambda x: len(x) > 0, before_text.split()))[-window:]
    before_window = " ".join(before_window_words)

    after_text = text[end_char:]
    after_window_words = list(filter(lambda x: len(x) > 0, after_text.split()))[:window]
    after_window = " ".join(after_window_words)

    return before_window, after_window


class GptNerTrainingGenerator:

    def generate(self, docs):
        return [
            {"prompt": self._create_prompt(doc), "completion": self._create_completion(doc)}
            for doc in docs
        ]

    @staticmethod
    def _create_prompt(doc):
        return f"{doc['text']} {GPT_PROMPT_END_INDICATOR}"

    @staticmethod
    def _create_completion(doc):
        chars_to_wrap = [(span['start'], span['end'], span['label']) for span in doc['spans'] if span['label'] in
                         SPAN_LABEL_TO_CHAR_WRAPPER]
        wrapped_chars = wrap_chars_with_overlaps(doc['text'], chars_to_wrap, _get_wrapped_citation)
        return f" {wrapped_chars}{GPT_COMPLETION_END_INDICATOR}"


class GptEntityClassificationTrainingGenerator:

    def __init__(self, before_wrapper="{{", after_wrapper="}}"):
        self.before_wrapper = before_wrapper
        self.after_wrapper = after_wrapper

    def generate(self, docs):
        data = []
        for doc in docs:
            for span in doc['spans']:
                data += [{
                    "prompt": self.create_prompt(doc['text'], span['start'], span['end']),
                    "completion": self._create_completion(doc, span)
                }]
        return data

    def create_prompt(self, text, start, end):
        before_window, after_window = get_window_around_match(start, end, text)
        span_text = text[start:end]
        wrapped_chars = f"{before_window} {self.before_wrapper}{span_text}{self.after_wrapper} {after_window}"
        return f"{wrapped_chars} {GPT_PROMPT_END_INDICATOR}"

    @staticmethod
    def _create_completion(doc, span):
        return f" {SPAN_LABEL_TO_CLASSICATION_TAG[span['label']]} {GPT_COMPLETION_END_INDICATOR}"


def get_gpt_training_data(task, docs):
    if task == "ner":
        generator = GptNerTrainingGenerator()
    elif task == "entity_classification":
        generator = GptEntityClassificationTrainingGenerator()
    else:
        raise Exception("Unrecognized task. Options are 'ner', 'entity_classification'.")
    return generator.generate(docs)


def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('input')
    parser.add_argument('task')
    parser.add_argument('training_filename')
    parser.add_argument('validation_filename')
    parser.add_argument('-m', '--db-host', dest='db_host')
    parser.add_argument('-p', '--db-port', dest='db_port', type=int)
    parser.add_argument("-u", "--user", dest="user", default="", nargs="?")
    parser.add_argument("-r", "--replicaset", dest="replicaset", default="", nargs="?")
    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    password = os.getenv('MONGO_PASSWORD')
    citation_docs = load_mongo_docs(args.input, args.db_host, args.db_port, args.user, password, args.replicaset)
    citation_docs = [doc for doc in citation_docs if doc['answer'] == 'accept']
    gpt_training = get_gpt_training_data(args.task, citation_docs)
    training_data, validation_data = train_test_split(gpt_training, random_state=613, train_size=0.8)
    print("TRAINING SIZE:", len(training_data))
    print("VALIDATION SIZE:", len(validation_data))
    srsly.write_jsonl(args.training_filename, training_data)
    srsly.write_jsonl(args.validation_filename, validation_data)
