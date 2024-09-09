from uuid import UUID
from abc import ABC, abstractmethod
from typing import Tuple
from typing import Iterable
from typing import Protocol
from typing import Iterator
from typing import Any
from torch import inference_mode
from dataclasses import dataclass, field

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer

type Batch = Tuple[Tensor, Tensor]

class Model(Module):
    def __init__(self, network: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.network = network
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, input: Tensor) -> Tensor:
        return self.network(input)

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return self.criterion(output, target)
    
    def fit(self, input: Tensor, target: Tensor) -> Tuple[Tensor, float]:
        self.optimizer.zero_grad()
        output = self(input)
        loss = self.loss(output, target)
        loss.backward()
        self.optimizer.step()
        return output, loss.item()
    
    @inference_mode()
    def evaluate(self, input: Tensor, target: Tensor) -> Tuple[Tensor, float]:
        output = self(input)
        loss = self.loss(output, target)
        return output, loss.item()

@dataclass
class Result:
    batch: int
    input: Tensor
    output: Tensor
    target: Tensor
    loss: float

@dataclass
class Metric:
    name: str
    history: list[Any] = field(default_factory=list)

class Loader[T](Protocol):
    def __iter__(self) -> Iterator[T]: ...

def train(model: Model, loader: Loader[Batch], device: str) -> Iterable[Result]:
    model.train()
    for batch, (input, target) in enumerate(loader, start=1):
        input, target = input.to(device), target.to(device)
        output, loss = model.fit(input, target)
        yield Result(batch, input, output, target, loss)

def evaluate(model: Model, loader: Loader[Batch], device: str) -> Iterable[Result]:
    model.eval()
    for batch, (input, target) in enumerate(loader, start=1):
        input, target = input.to(device), target.to(device)
        output, loss = model.evaluate(input, target)
        yield Result(batch, input, output, target, loss)

class Experiment:
    def __init__(self, id: UUID, name: str):
        super().__init__()
        self.id = id
        self.name = name
        self.epochs = 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Experiment):
            return False
        return self.id == other.id and self.name == other.name
    
    def __hash__(self) -> int:
        return hash((self.id, self.name))