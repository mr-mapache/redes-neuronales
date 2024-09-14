from abc import ABC, abstractmethod
from uuid import UUID
from typing import Protocol
from typing import Tuple
from typing import Any
from typing import Callable
from typing import Iterator
from typing import Optional
from typing import TypedDict
from enum import StrEnum
from dataclasses import dataclass, field

class Tensor(Protocol):
    def to(self, device: str) -> Any: ...

class Model(Protocol):
    nn: Callable
    criterion: Optional[Callable]
    optimizer: Optional[Callable]

    def fit(self, input: Tensor, target: Tensor) -> Tuple[Tensor, float]: ...	

    def evaluate(self, input: Tensor, target: Tensor) -> Tuple[Tensor, float]: ...

    def train(self) -> None: ...

    def eval(self) -> None: ...

class Phase(StrEnum):
    START = 'start'
    TRAIN = 'train'
    EVALUATION = 'evaluation'
    BREAK = 'break'

@dataclass(slots=True)
class Result:
    batch: int
    input: Tensor
    target: Tensor
    output: Tensor
    loss: float
    phase: Phase

class Callback(Protocol):
    def __call__(self, message: Result) -> None: ...

class Loader(Protocol):
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]: ...

@dataclass(slots=True)
class Metric(ABC):
    history: dict[Phase, list] | None = field(default_factory=lambda: {
        Phase.TRAIN: [],
        Phase.EVALUATION: []
    })
    name: str = field(default='metric')

    @abstractmethod
    def __call__(self, batch: int, *args: Any) -> Any: ...

@dataclass
class State:
    nn: str
    optimizer: str
    criterion: str
    batch_size: int
    epochs: int

@dataclass
class Experiment:
    id: UUID
    name: str

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, Experiment):
            return False
        return self.id == value.id and self.name == value.name
    
    def __hash__(self) -> int:
        return hash(self.id)