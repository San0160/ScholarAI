from abc import ABC, abstractmethod


class BaseGenerator(ABC):

    @abstractmethod
    def generate(
        self,
        messages: list[dict]
    ) -> str:
        pass