import django
django.setup()
from sefaria.model import *
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_community.adapters.openai import convert_message_to_dict
from srsly import write_jsonl
from sklearn.model_selection import train_test_split

def get_prompts_with_diffs():
    prompts_with_diffs = []
    for link in RefTopicLinkSet({"descriptions.en.ai_prompt": {"$exists": True}}):
        description = link.descriptions['en']
        if description['ai_prompt'] != description['prompt']:
            prompts_with_diffs += [(description['ai_prompt'], description['prompt'])]
    return prompts_with_diffs


class GptPromptTrainingGenerator:

    @staticmethod
    def generate(input_toprompt, gold_standard_toprompt=None):
        """
        Generate a list of messages to feed to GPT to either train or run on
        :return:
        """
        example = GptPromptTrainingGenerator.generate_one(input_toprompt, gold_standard_toprompt)
        return GptPromptTrainingGenerator.serialize_messages(example)

    @staticmethod
    def generate_one(input_toprompt, gold_standard_toprompt=None):
        return GptPromptTrainingGenerator._generate_one_chat_format(input_toprompt, gold_standard_toprompt)

    @staticmethod
    def _generate_one_chat_format(input_toprompt, gold_standard_toprompt=None):
        messages = [
            SystemMessage(content=GptPromptTrainingGenerator._create_system_prompt()),
            HumanMessage(content=GptPromptTrainingGenerator._create_prompt(input_toprompt)),
        ]
        if gold_standard_toprompt:
            messages += [AIMessage(content=GptPromptTrainingGenerator._create_completion(gold_standard_toprompt))]
        return messages

    @staticmethod
    def serialize_messages(messages):
        return {"messages": [convert_message_to_dict(message) for message in messages]}

    @staticmethod
    def _create_system_prompt():
        return "You are Jewish scholar knowledgeable in all Torah texts. Your goal is to take a description of a Jewish source and rewrite it, adhering religiously to your style guide."

    @staticmethod
    def _create_prompt(input_toprompt):
        return input_toprompt

    @staticmethod
    def _create_completion(gold_standard_toprompt):
        return gold_standard_toprompt


def save_fine_tune_training_set(prompts_with_diffs):
    write_jsonl("output/fine_tune_training_set.jsonl", training_set)


if __name__ == '__main__':
    prompts_with_diffs = get_prompts_with_diffs()
    fine_tune_data = [GptPromptTrainingGenerator.generate(a, b) for (a, b) in prompts_with_diffs]
    training_data, validation_data = train_test_split(fine_tune_data, random_state=613, train_size=0.999)
    write_jsonl("output/fine_tune_training_set.jsonl", training_data)
    write_jsonl("output/fine_tune_validation_set.jsonl", validation_data)