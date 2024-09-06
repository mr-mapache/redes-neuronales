from torch import Tensor
from torch import chunk
from torch.nn import Module
from torch.nn import Linear, ReLU
from torch.nn import Dropout

class MLP(Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int, p: float = 0.5, activation: Module = ReLU()):
        super().__init__()
        self.activation = activation
        self.dropout = Dropout(p)
        self.input_layer = Linear(input_dimension, hidden_dimension)
        self.output_layer = Linear(hidden_dimension, output_dimension)

    def forward(self, input: Tensor) -> Tensor:
        output = self.input_layer(input)
        output = self.activation(output)
        output = self.dropout(output)
        return self.output_layer(output)

class GLU(Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, p: float = 0.5, activation: Module = ReLU()):
        super().__init__()
        self.activation = activation
        self.dropout = Dropout(p)
        self.input_layer = Linear(input_dimension, 2* hidden_dimension)
        self.output_layer = Linear(hidden_dimension, input_dimension)

    def forward(self, input: Tensor) -> Tensor:
        output = self.input_layer(input)
        output, gate = chunk(output, 2, dim=-1)
        output = self.activation(output)
        output = self.dropout(output)
        output = output * gate
        return self.output_layer(output)