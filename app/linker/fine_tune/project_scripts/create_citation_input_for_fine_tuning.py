from dataclasses import dataclass, asdict
import os
import argparse
from sefaria.utils.util import wrap_chars_with_overlaps
from sklearn.model_selection import train_test_split
import srsly
from util.sefaria_specific import load_mongo_docs
from linker.fine_tune.project_scripts import constants
from langchain.chat_models.openai import convert_message_to_dict
from langchain.schema import HumanMessage, SystemMessage, AIMessage


SPAN_LABEL_TO_CLASSICATION_TAG = {
    "Person": "Person",
    "Group-of-People": "Group",
    "Group": "Group",
    "Name-of-God": "Person",
    "Citation": "Citation",
}


def _get_wrapped_citation(citation, metadata):
    wrapped = f"{constants.ENTITY_PRE_WRAPPER}{citation}{constants.ENTITY_POST_WRAPPER}"
    len_pre = len(constants.ENTITY_PRE_WRAPPER)
    len_post = len(constants.ENTITY_POST_WRAPPER)
    return wrapped, len_pre, len_post


def get_window_around_match(start_char:int, end_char:int, text:str, window:int=10) -> tuple:
    before_text = text[:start_char]
    before_window_words = list(filter(lambda x: len(x) > 0, before_text.split()))[-window:]
    before_window = " ".join(before_window_words)

    after_text = text[end_char:]
    after_window_words = list(filter(lambda x: len(x) > 0, after_text.split()))[:window]
    after_window = " ".join(after_window_words)

    return before_window, after_window


class GptNerTrainingGenerator:

    format = "completion"

    @staticmethod
    def generate(docs, is_labeled=True):
        """
        Generate a list of messages to feed to GPT to either train or run on
        :param docs: input data in the form of spaCy docs
        :param is_labeled: are the docs labeled with the correct answers. False for inference use.
        :return:
        """
        examples = [GptNerTrainingGenerator.generate_one(doc, is_labeled) for doc in docs]
        if GptNerTrainingGenerator.format == "completion":
            return examples
        return [GptNerTrainingGenerator.serialize_messages(example) for example in examples]

    @staticmethod
    def generate_one(doc, is_labeled=True):
        if GptNerTrainingGenerator.format == "completion":
            return GptNerTrainingGenerator._generate_one_completion_format(doc, is_labeled)
        return GptNerTrainingGenerator._generate_one_chat_format(doc, is_labeled)

    @staticmethod
    def _generate_one_chat_format(doc, is_labeled=True):
        messages = [
            SystemMessage(content=GptNerTrainingGenerator._create_system_prompt()),
            HumanMessage(content=GptNerTrainingGenerator._create_prompt(doc)),
        ]
        if is_labeled:
            messages += [AIMessage(content=GptNerTrainingGenerator._create_completion(doc))]
        return messages

    @staticmethod
    def _generate_one_completion_format(doc, is_labeled=False):
        prompt = GptNerTrainingGenerator._create_prompt(doc)
        completion = GptNerTrainingGenerator._create_completion(doc)
        prompt_formatted = f"{prompt} {constants.GPT_PROMPT_END_INDICATOR}"
        completion_formatted = f"{completion}{constants.GPT_COMPLETION_END_INDICATOR}"
        if is_labeled:
            return {
                "prompt": prompt_formatted,
                "completion": completion_formatted,
            }
        return prompt_formatted

    @staticmethod
    def serialize_messages(messages):
        return {"messages": [convert_message_to_dict(message) for message in messages]}

    @staticmethod
    def _create_system_prompt():
        return "You are Jewish scholar knowledgeable in all Torah texts. Your task is to wrap all people, groups of " \
               "people and citations in double curly braces."

    @staticmethod
    def _create_prompt(doc):
        return doc['text']

    @staticmethod
    def _create_completion(doc):
        chars_to_wrap = [(span['start'], span['end'], None) for span in doc['spans'] if span['label'] in
                         SPAN_LABEL_TO_CLASSICATION_TAG]
        return wrap_chars_with_overlaps(doc['text'], chars_to_wrap, _get_wrapped_citation)


class GptEntityClassificationTrainingGenerator:

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
        wrapped_chars = f"{before_window} {constants.ENTITY_PRE_WRAPPER}{span_text}{constants.ENTITY_POST_WRAPPER} {after_window}"
        return f"{wrapped_chars} {constants.GPT_PROMPT_END_INDICATOR}"

    @staticmethod
    def _create_completion(doc, span):
        return f" {SPAN_LABEL_TO_CLASSICATION_TAG[span['label']]} {constants.GPT_COMPLETION_END_INDICATOR}"


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
    citation_docs = [doc for doc in citation_docs if doc.get('answer', 'accept') == 'accept']
    gpt_training = get_gpt_training_data(args.task, citation_docs)
    training_data, validation_data = train_test_split(gpt_training, random_state=613, train_size=0.8)
    print("TRAINING SIZE:", len(training_data))
    print("VALIDATION SIZE:", len(validation_data))
    srsly.write_jsonl(args.training_filename, training_data)
    srsly.write_jsonl(args.validation_filename, validation_data)
