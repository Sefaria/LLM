from typing import Any, Callable

class Artifact:

    def __init__(self, data: Any):
        self._data = data

    def pipe(self, func: Callable, *args, **kwargs):
        result = func(self._data, *args, **kwargs)
        return Artifact(result)

    @property
    def data(self) -> Any:
        return self._data