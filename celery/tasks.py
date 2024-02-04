from celery import Celery
from topic_prompt.topic_prompt_generator import get_toprompts
from sefaria_interface.topic_prompt_input import TopicPromptInput

app = Celery('llm')
app.config_from_object('celery_config')


@app.task()
def generate_topic_prompts(raw_topic_prompt_input: dict):
    tp_input = TopicPromptInput.create(raw_topic_prompt_input)
    return get_toprompts(tp_input)
