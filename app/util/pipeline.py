from typing import Any, Callable

class Artifact:

    def __init__(self, data: Any):
        self._data = data

    def __rshift__(self, other: Callable):
        result = other(self._data)
        return Artifact(result)