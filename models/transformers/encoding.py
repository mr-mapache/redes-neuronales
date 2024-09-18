import math
from torch import randn
from torch import exp, arange, outer, ones_like
from torch import view_as_complex, view_as_real, polar
from torch import Tensor
from torch.nn import Module
from torch.nn import Parameter

class Encoding(Module):
    ...

class Learnable(Encoding):
    def __init__(self, model_dimension: int, sequence_lenght_limit: int = 196):
        super().__init__()
        self.sequence_lenght_limit = sequence_lenght_limit
        self.position_embeddings = Parameter(randn(1, sequence_lenght_limit + 1, model_dimension))

    def forward(self, input: Tensor) -> Tensor:
        assert input.size(1) <= self.sequence_lenght_limit + 1, 'input sequence is too long'
        input = input + self.position_embeddings[:, :input.size(1)]
        return input

class Rotatory(Encoding):
    def __init__(self, model_dimension: int, sequence_lenght_limit: int, scaling_factor: float = 10000.0):
        super().__init__()
        frequencies = Tensor(sequence_lenght_limit + 1, model_dimension // 2)
        frequencies = exp(- arange(0, model_dimension, 2) * math.log(scaling_factor) / model_dimension)
        frequencies = outer(arange(sequence_lenght_limit + 1), frequencies)
        self.embeddings = polar(ones_like(frequencies), frequencies)

    def forward(self, sequence: Tensor, start_position: int = 0) -> Tensor:
        batch_size, number_of_heads, sequence_lenght, heads_dimension = sequence.shape
        assert heads_dimension % 2 == 0, 'The heads dimension must be divisible by 2'
        sequence = sequence.view(batch_size, number_of_heads, sequence_lenght, heads_dimension // 2, 2)
        sequence = view_as_complex(sequence)
        sequence = sequence * self.embeddings[start_position : start_position + sequence_lenght, :heads_dimension //2].to(sequence.device)
        sequence = view_as_real(sequence)
        return sequence.view(batch_size, number_of_heads, sequence_lenght, heads_dimension)

class LearnableRotatory(Encoding):
    def __init__(self, model_dimension: int, sequence_lenght_limit: int, scaling_factor: float = 10000.0):
        super().__init__()
        frequencies = Tensor(sequence_lenght_limit + 1, model_dimension // 2)
        frequencies = exp(- arange(0, model_dimension, 2) * math.log(scaling_factor) / model_dimension)
        frequencies = outer(arange(sequence_lenght_limit + 1), frequencies)
        self.embeddings = Parameter(polar(ones_like(frequencies), frequencies), requires_grad=True)

    def forward(self, sequence: Tensor, start_position: int = 0) -> Tensor:
        batch_size, number_of_heads, sequence_lenght, heads_dimension = sequence.shape
        assert heads_dimension % 2 == 0, 'The heads dimension must be divisible by 2'
        sequence = sequence.view(batch_size, number_of_heads, sequence_lenght, heads_dimension // 2, 2)
        sequence = view_as_complex(sequence)
        sequence = sequence * self.embeddings[start_position : start_position + sequence_lenght, :heads_dimension //2].to(sequence.device)
        sequence = view_as_real(sequence)
        return sequence.view(batch_size, number_of_heads, sequence_lenght, heads_dimension)