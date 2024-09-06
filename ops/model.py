from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch import Tensor

class Model(Module):
    def __init__(self, module: Module, criterion: Module, optimizer: Optimizer):
        super().__init__()
        self.module = module
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, input: Tensor) -> Tensor:
        return self.module(input)

    def loss(self, output: Tensor, target: Tensor) -> Tensor:
        return self.criterion(output, target)    