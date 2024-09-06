from abc import ABC, abstractmethod
from ops.model import Model
from typing import Any
from typing import Protocol
from typing import Iterator, Tuple
from torch import Tensor


class Publisher(Protocol):
    def publish(self, topic: str, message: Any) -> None:
        ...


class Data(Protocol):
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        ...


class Consumer(ABC):
    @abstractmethod
    def consume(self, message: Any) -> None: ...

    def stop(self) -> None: 
        pass


class Repository(ABC):
    @abstractmethod
    def create(self, *args, **kwargs) -> Model:
        pass

    @abstractmethod
    def save(self, model: Model):
        pass

    @abstractmethod
    def restore(self, model: Model):
        pass

    @abstractmethod
    def delete(self, name: str):
        pass