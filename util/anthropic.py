import anthropic
import os


def get_completion_anthropic(prompt, max_tokens_to_sample: int = 1000000, verbose=True):
    c = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = c.completions.create(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt} {anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        model="claude-instant-1",
        max_tokens_to_sample=max_tokens_to_sample,
        temperature=0,
    )
    if verbose:
        print("NUM INPUT TOKENS", c.count_tokens(prompt))
        print("STOP REASON", resp.stop_reason)
    return resp.completion
