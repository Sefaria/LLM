from typing import List
from celery import Celery
from topic_prompt.topic_prompt_generator import get_toprompts
from sefaria_interface.topic_prompt_input import TopicPromptInput

app = Celery('llm')
app.config_from_object('celery.config')


@app.task()
def generate_topic_prompts(raw_topic_prompt_input: dict) -> dict:
    tp_input = TopicPromptInput.create(raw_topic_prompt_input)
    toprompt_options_list = get_toprompts(tp_input)
    # only return the first option for now
    return {
        "lang": tp_input.lang,
        "prompts": [
           options.toprompts[0].serialize() for options in toprompt_options_list
        ]
    }
