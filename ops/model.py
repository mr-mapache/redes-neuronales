from typing import Any
from typing import Tuple
from typing import Optional

from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch import inference_mode

class Model(Module):
    def __init__(self, name: str, network: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.epochs = 0
        self.name = name
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