from enum import Enum


class LLMCompany(Enum):

    OPENAI = "OpenAI"
    ANTHROPIC = "Anthropic"


class AbstractMessage:

    ROLE_MAP = {
        LLMCompany.ANTHROPIC: {
            "human": "user",
            "system": "system",
            "ai": "assistant",
        },
        LLMCompany.OPENAI: {
            "human": "user",
            "system": "system",
            "ai": "assistant",
        }
    }

    def __init__(self, role, content):
        self.role = role
        self.content = content

    def _get_converted_role(self, company: LLMCompany) -> str:
        return self.ROLE_MAP[company][self.role]


    def serialize(self, company: LLMCompany) -> dict:
        return {
            "role": self._get_converted_role(company),
            "content": self.content
        }


class HumanMessage(AbstractMessage):

    def __init__(self, content):
        super().__init__("human", content)


class AIMessage(AbstractMessage):

    def __init__(self, content: str):
        super().__init__("ai", content)


class SystemMessage(AbstractMessage):

    def __init__(self, content):
        super().__init__("system", content)
