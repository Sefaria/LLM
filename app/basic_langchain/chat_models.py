from typing import List
from abc import ABC, abstractmethod
from anthropic import Anthropic, InternalServerError, RateLimitError
from time import sleep
from openai import OpenAI
from basic_langchain.schema import AIMessage, AbstractMessage, LLMCompany
from basic_langchain.cache import sqlite_cache


class AbstractChatModel(ABC):

    COMPANY = None

    def __init__(self, model, temperature, **kwargs):
        self.model = model
        self.temperature = temperature

    def _serialize_messages(self, messages: List[AbstractMessage]) -> List[dict]:
        return [m.serialize(self.COMPANY) for m in messages]

    @abstractmethod
    def __call__(self, messages: List[AbstractMessage]) -> AIMessage:
        raise NotImplementedError


class ChatOpenAI(AbstractChatModel):

    COMPANY = LLMCompany.OPENAI

    def __init__(self, model, temperature):
        super().__init__(model, temperature)
        self.client = OpenAI()

    @sqlite_cache('chat')
    def __call__(self, messages: List[AbstractMessage]) -> AIMessage:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=self._serialize_messages(messages)
        )
        text = response.choices[0].message.content
        return AIMessage(text)


class ChatAnthropic(AbstractChatModel):

    COMPANY = LLMCompany.ANTHROPIC

    def __init__(self, model, temperature, max_tokens=4096):
        super().__init__(model, temperature, max_tokens=max_tokens)
        self.client = Anthropic()
        self.max_tokens = max_tokens

    @sqlite_cache('chat')
    def __call__(self, messages: list[AbstractMessage]) -> AIMessage:
        system = "You are a helpful AI."
        if len(messages) > 0 and messages[0].role == "system":
            # claude wants system messages as a kwarg
            system = messages[0].content
            messages.pop(0)
        response = self._api_call(system, messages)
        text = response.content[0].text
        return AIMessage(text)

    def _api_call(self, system, messages: list[AbstractMessage]):
        try:
            return self.client.messages.create(
                model=self.model,
                system=system,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=self._serialize_messages(messages)
            )
        except InternalServerError:
            print("Internal Server Error. Waiting 5 seconds...")
            sleep(5)
            return self._api_call(system, messages)
        except RateLimitError:
            print("Rate Limit Error. Waiting 10min...")
            sleep(60*10)
            return self._api_call(system, messages)

