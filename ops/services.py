from ops.model import Model
from ops.ports import Data, Publisher
from torch import Tensor, argmax
from dataclasses import dataclass

@dataclass
class Result:
    output: Tensor
    target: Tensor
    loss: float

from logging import getLogger
logger = getLogger(__name__)

def train(model: Model, data: Data, publisher: Publisher, device: str):
    model.train()
    for input, target in data:
        input, target = input.to(device), target.to(device)
        output, loss = model.fit(input, target)
        publisher.publish('train-results', Result(output, target, loss))
        
    publisher.publish('train-results', model.epochs)
        
def test(model: Model, data: Data, publisher: Publisher, device: str):
    model.eval()
    for input, target in data:
        input, target = input.to(device), target.to(device)
        output, loss = model.evaluate(input, target)
        publisher.publish('test-results', Result(output, target, loss))
    publisher.publish('test-results',  model.epochs)

def accuracy(predictions: Tensor, target: Tensor) -> float:
    return (predictions == target).float().mean().item()

def predictions(output: Tensor) -> Tensor:
    return argmax(output, dim=1)