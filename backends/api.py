"""Pay-per-token backends. Set ANTHROPIC_API_KEY or OPENAI_API_KEY."""
from backends import SYSTEM_PROMPT


class ClaudeAPI:
    def __init__(self, model, effort, web_search, refusal_fallback=True, max_tokens=64000):
        import anthropic

        self.client = anthropic.Anthropic(max_retries=8)
        self.request = {
            "model": model, "max_tokens": max_tokens, "system": SYSTEM_PROMPT,
            "thinking": {"type": "adaptive"}, "output_config": {"effort": effort},
            "cache_control": {"type": "ephemeral"},
        }
        if web_search:
            self.request["tools"] = [{"type": "web_search_20260209", "name": "web_search"}]
        if refusal_fallback:   # on a refusal, another model continues; transcript "info" shows which
            self.request.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
        self.messages = []
        self.info = {}

    def send(self, text):
        self.messages.append({"role": "user", "content": text})
        pieces = []
        while True:
            with self.client.beta.messages.stream(messages=self.messages, **self.request) as stream:
                response = stream.get_final_message()
            self.messages.append({"role": "assistant", "content": response.content})
            pieces += [block.text for block in response.content if block.type == "text"]
            if response.stop_reason != "pause_turn":   # a long web search pauses; resend to continue
                break
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Request refused: {response.stop_details}")
        self.info = {"served_by": response.model, "usage": response.usage.model_dump(exclude_none=True)}
        return "".join(pieces)


class OpenAIAPI:
    def __init__(self, model, effort, web_search):
        import openai

        self.client = openai.OpenAI(max_retries=8)
        self.request = {"model": model, "instructions": SYSTEM_PROMPT, "reasoning": {"effort": effort}}
        if web_search:
            self.request["tools"] = [{"type": "web_search"}]
        self.previous_id = None
        self.info = {}

    def send(self, text):
        with self.client.responses.stream(
                input=text, previous_response_id=self.previous_id, **self.request) as stream:
            response = stream.get_final_response()
        self.previous_id = response.id
        self.info = {"served_by": response.model, "usage": response.usage.model_dump()}
        return response.output_text
