from torch import Tensor
from torch import chunk
from torch.nn import Module, Flatten
from torch.nn import Linear, ReLU, GELU, SiLU
from torch.nn import Dropout
from models.dropconnect import DropConnectLinear, DropConnectBatchAverage

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

class MLP(Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int, p: float = 0.5, activation: str = 'relu'):
        super().__init__()
        self.flatten = Flatten(start_dim=1)
        self.activation = select(activation)
        self.dropout = Dropout(p)
        self.input_layer = Linear(input_dimension, hidden_dimension)
        self.output_layer = Linear(hidden_dimension, output_dimension)

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = self.input_layer(sequence.flatten(1))
        sequence = self.activation(sequence)
        sequence = self.dropout(sequence)
        return self.output_layer(sequence)

class GLU(Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int, p: float = 0.5, activation: str = 'relu'):
        super().__init__()
        self.hidden_dimension = int(hidden_dimension * 2 / 3)
        self.activation = select(activation)
        self.dropout = Dropout(p)
        self.input_layer = Linear(input_dimension, 2* self.hidden_dimension, bias=False)
        self.output_layer = Linear(self.hidden_dimension, output_dimension)

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = self.input_layer(sequence.flatten(1))
        sequence, gate = chunk(sequence, 2, dim=-1)
        sequence = self.activation(sequence)
        sequence = self.dropout(sequence)
        sequence = sequence * gate
        return self.output_layer(sequence)
    
class DCP(Module):
    def __init__(self, input_dimension: int, hidden_dimension: int, output_dimension: int, p: float = 0.5, activation: str = 'relu'):
        super().__init__()
        self.input_layer = DropConnectLinear(input_dimension, hidden_dimension, p=p, max_batch_size=1024)
        self.activation = select(activation)
        self.output_layer = Linear(hidden_dimension, output_dimension)
        self.batch_average = DropConnectBatchAverage()

    def forward(self, sequence: Tensor) -> Tensor:
        sequence = self.input_layer(sequence.flatten(1))
        sequence = self.activation(sequence)
        sequence = self.batch_average(sequence)
        return self.output_layer(sequence)