from torch import Tensor
from torch.nn import Module, Linear, Sequential
from torch.nn import ReLU
from torch.nn import Dropout

class MLP(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, bias: bool = True, p: float = 0.5):
        super().__init__()
        self.input_layer = Linear(model_dimension, hidden_dimension, bias=bias)
        self.activation = ReLU()
        self.dropout = Dropout(p)
        self.output_layer = Linear(hidden_dimension, model_dimension, bias=False)
       
    def forward(self, input: Tensor) -> Tensor:
        output = self.activation(self.input_layer(input))
        output = self.dropout(output)
        return self.output_layer(output)

class GLU(Module):
    def __init__(self, model_dimension: int, hidden_dimension: int, bias: bool = True):
        super().__init__()
        if not hidden_dimension % 3 == 0:
            print("Warning: hidden_dimension should be a multiple of 3 for having the same number of parameters as a regular MLP")
        self.hidden_dimension = (hidden_dimension * 2) // 3
        self.gate_dimension = hidden_dimension 

        self.activation = ReLU()
        self.input_layer = Linear(model_dimension, self.hidden_dimension, bias=bias)
        self.output_layer = Linear(self.hidden_dimension, hidden_dimension, bias=False)
        self.gate_layer = Linear(model_dimension, self.hidden_dimension, bias=False)
       
    def forward(self, input: Tensor) -> Tensor:
        output = self.activation(self.input_layer(input)) * self.gate_layer(input)
        return self.output_layer(output)