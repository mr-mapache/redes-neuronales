from torch import Tensor
from torch import chunk
from torch.nn import Module, Flatten
from torch.nn import Linear, ReLU, GELU, SiLU
from torch.nn import Dropout

def select(activation: str) -> Module:
    match activation:
        case 'relu':
            return ReLU()
        case 'gelu':
            return GELU()
        case 'silu':
            return SiLU()
        case _:
            raise ValueError(f'Activation {activation} not supported')
        
class FFN(Module):
    ...

class MLP(FFN):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int, p: float = 0.5, activation: str = 'relu'):
        super().__init__()
        self.flatten = Flatten(start_dim=1)
        self.activation = select(activation)
        self.dropout = Dropout(p)
        self.input_layer = Linear(input_dimension, hidden_dimension)
        self.output_layer = Linear(hidden_dimension, output_dimension)

    def forward(self, input: Tensor) -> Tensor:
        output = self.input_layer(input)
        output = self.activation(output)
        output = self.dropout(output)
        return self.output_layer(output)

class GLU(FFN):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int, p: float = 0.5, activation: str = 'relu'):
        super().__init__()
        hidden_dimension = int(hidden_dimension * 2 / 3)
        self.activation = select(activation)
        self.dropout = Dropout(p)
        self.input_layer = Linear(input_dimension, 2* hidden_dimension)
        self.output_layer = Linear(hidden_dimension, output_dimension)

    def forward(self, input: Tensor) -> Tensor:
        output = self.input_layer(input)
        output, gate = chunk(output, 2, dim=-1)
        output = self.activation(output)
        output = self.dropout(output)
        output = output * gate
        return self.output_layer(output)