from dataclasses import dataclass
from typing import Any
from typing import Iterator, Tuple
from typing import Protocol
from torch import Tensor
from torch import argmax
from torch import inference_mode
from ops.model import Model

class Publisher(Protocol):
    def publish(self, topic: str, payload: Any) -> None:
        ...

class Data(Protocol):
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        ...
    
@dataclass
class Result:
    batch: int
    loss: float
    output: Tensor
    target: Tensor    

def train(model: Model, data: Data, publisher: Publisher, device: str):
    model.module.train()
    for batch, (input, target) in enumerate(data, start=1):
        input, target = input.to(device), target.to(device)
        model.optimizer.zero_grad()
        output = model(input)
        loss = model.loss(output, target)
        loss.backward()
        model.optimizer.step()
        publisher.publish('train', Result(batch, loss.item(), output, target))
        
@inference_mode()
def test(model: Model, data: Data, publisher: Publisher, device: str):
    model.module.eval()
    for batch, (input, target) in enumerate(data, start=1):
        input, target = input.to(device), target.to(device)
        output = model(input)
        loss = model.loss(output, target)
        publisher.publish('test', Result(batch, loss.item(), output, target))

def accuracy(predictions: Tensor, target: Tensor) -> float:
    return (predictions == target).float().mean().item()

def predictions(output: Tensor) -> Tensor:
    return argmax(output, dim=1)