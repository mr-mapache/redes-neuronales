from torch import Tensor, argmax
from src.domain.models import Metric, Phase
from typing import Any

def accuracy(predictions: Tensor, target: Tensor) -> float:
    return (predictions == target).float().mean().item()

def predictions(output: Tensor) -> Tensor:
    return argmax(output, dim=1)

class Accuracy(Metric):
    def __init__(self, history: dict[Phase, list] | None = None):
        super().__init__('accuracy', history)
        self.average = 0.0
        self.batch = 0
        self.phase = Phase.TRAIN
    
    def __call__(self, batch: int, output: Tensor, target: Tensor, phase: Phase) -> float:
        if phase == Phase.BREAK:
            self.history[self.phase].append(self.average)
            self.phase = Phase.TRAIN if self.phase == Phase.EVALUATION else Phase.EVALUATION
            self.average = 0.0
        else:
            self.batch = batch
            self.average = (self.average * (batch - 1) + accuracy(predictions(output), target)) / batch
            return self.average

class Loss(Metric):
    def __init__(self, history: dict[Phase, list] | None = None):
        super().__init__('loss', history)
        self.average = 0.0
        self.batch = 0
        self.phase = Phase.TRAIN
    
    def __call__(self, batch: int, loss: float, phase: Phase) -> float:
        if phase == Phase.BREAK:
            self.history[self.phase].append(self.average)
            self.phase = Phase.TRAIN if self.phase == Phase.EVALUATION else Phase.EVALUATION
            self.average = 0.0
        else:
            self.batch = batch
            self.average = (self.average * (batch - 1) + loss) / batch
            return self.average

def factory(name: str, history: dict[Phase, list] | None = None) -> Metric:
    match name:
        case 'accuracy':
            return Accuracy(history)
        case 'loss':
            return Loss(history)
        case _:
            raise ValueError(f'Unknown metric: {name}')