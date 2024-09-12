from dataclasses import dataclass, field
from torch import Tensor, argmax
from src.domain.models import Metric, Phase
from typing import Any

def accuracy(predictions: Tensor, target: Tensor) -> float:
    return (predictions == target).float().mean().item()

def predictions(output: Tensor) -> Tensor:
    return argmax(output, dim=1)

@dataclass(slots=True, kw_only=True)
class Accuracy(Metric):
    name: str = field(default='accuracy', init=False)
    average: float = field(default=0.0, init=False)
    batch: int = field(default=0, init=False)
    phase: Phase = field(default=Phase.TRAIN)
    
    def __call__(self, batch: int, output: Tensor, target: Tensor, phase: Phase) -> float:
        if phase == Phase.BREAK:
            self.history[self.phase].append(self.average)
            self.phase = Phase.TRAIN if self.phase == Phase.EVALUATION else Phase.EVALUATION
            self.average = 0.0
        else:
            self.batch = batch
            self.average = (self.average * (batch - 1) + accuracy(predictions(output), target)) / batch
            return self.average

@dataclass(slots=True, kw_only=True)
class Loss(Metric):
    name: str = field(default='loss', init=False)    
    average: float = field(default=0.0, init=False)
    batch: int = field(default=0, init=False)
    phase: Phase = field(default=Phase.TRAIN)


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
            return Accuracy(history=history or { Phase.TRAIN: [], Phase.EVALUATION: [] })
        case 'loss':
            return Loss(history=history or { Phase.TRAIN: [], Phase.EVALUATION: [] })
        case _:
            raise ValueError(f'Unknown metric: {name}')