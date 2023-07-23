import anthropic
import os

async def get_completion_anthropic(prompt, max_tokens_to_sample: int = 100):
    c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])
    resp = await c.acompletion(
        prompt=prompt,
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-v1",
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=0,
    )
    print("STOP REASON", resp['stop_reason'])
    return resp['completion']
