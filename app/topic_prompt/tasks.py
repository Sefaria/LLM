from celery import shared_task
from topic_prompt.topic_prompt_generator import get_toprompts
from sefaria_llm_interface.topic_prompt import TopicPromptInput, TopicPrompt, TopicPromptGenerationOutput
from dataclasses import asdict


@shared_task(name='llm.generate_topic_prompts')
def generate_topic_prompts(raw_topic_prompt_input: dict) -> dict:
    tp_input = TopicPromptInput(**raw_topic_prompt_input)
    toprompt_options_list = get_toprompts(tp_input)
    # only return the first option for now
    toprompts = [TopicPrompt(**options.toprompts[0].serialize()) for options in toprompt_options_list]
    output = TopicPromptGenerationOutput(lang=tp_input.lang, prompts=toprompts)
    return asdict(output)
