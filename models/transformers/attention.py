import math
from typing import Optional
from torch import Tensor
from torch import randn
from torch import exp, arange, outer, ones_like
from torch import view_as_complex, view_as_real, polar
from torch.nn import Module
from torch.nn import Linear
from torch.nn import Parameter
from torch.nn.functional import scaled_dot_product_attention

def split_heads(sequence: Tensor, number_of_heads: int) -> Tensor:
    batch_size, sequence_length, model_dimension = sequence.shape
    sequence = sequence.view(batch_size, sequence_length, number_of_heads, model_dimension // number_of_heads)
    sequence = sequence.transpose(1, 2)
    assert sequence.dim() == 4
    return sequence

def concat_heads(sequence: Tensor) -> Tensor:
    batch_size, number_of_heads, sequence_lenght, heads_dimension = sequence.shape
    sequence = sequence.transpose(1, 2)
    sequence = sequence.reshape(batch_size, sequence_lenght, heads_dimension* number_of_heads)
    return sequence

def precompute_rotatory_embeddings(model_dimension: int, sequence_lenght_limit: int, scaling_factor: float = 10000.0) -> Tensor:
    frequencies = Tensor(sequence_lenght_limit, model_dimension // 2)
    frequencies = exp(- arange(0, model_dimension, 2) * math.log(scaling_factor) / model_dimension)
    frequencies = outer(arange(sequence_lenght_limit), frequencies)
    return polar(ones_like(frequencies), frequencies)

def apply_rotatory_embeddings(sequence: Tensor, rotatory_embeddings: Tensor) -> Tensor:
    batch_size, number_of_heads, sequence_lenght, heads_dimension = sequence.shape
    assert heads_dimension % 2 == 0, 'The heads dimension must be divisible by 2'
    sequence = sequence.view(batch_size, number_of_heads, sequence_lenght, heads_dimension // 2, 2)
    sequence = view_as_complex(sequence)
    sequence = sequence * rotatory_embeddings
    sequence = view_as_real(sequence)
    return sequence.view(batch_size, number_of_heads, sequence_lenght, heads_dimension)


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
    

class Attention(Module):
    ...

class MultiheadAttention(Attention):
    def __init__(self, model_dimension: int, number_of_heads: int, number_of_kv_heads: int):
        super().__init__()
        self.model_dimension = model_dimension
        self.heads_dimension = model_dimension // number_of_heads
        self.number_of_heads = number_of_heads
        self.number_of_kv_heads = number_of_kv_heads
        self.repeats = self.number_of_heads // self.number_of_kv_heads

        self.q_projector = Linear(model_dimension, self.heads_dimension * self.number_of_heads, bias=False)
        self.k_projector = Linear(model_dimension, self.heads_dimension * self.number_of_kv_heads, bias=False)
        self.v_projector = Linear(model_dimension, self.heads_dimension * self.number_of_kv_heads, bias=False)
        self.output_projector = Linear(self.number_of_heads * self.heads_dimension, model_dimension, bias=False)
        
    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        query, key, value = self.q_projector(sequence), self.k_projector(sequence), self.v_projector(sequence)        
        query, key, value = split_heads(query, self.number_of_heads), split_heads(key, self.number_of_kv_heads), split_heads(value, self.number_of_kv_heads)
        attention = scaled_dot_product_attention(query, key, value, mask)
        attention = concat_heads(attention)
        return self.output_projector(attention)

class RopeMultiheadAttention(Attention):
    def __init__(self, model_dimension: int, number_of_heads: int, number_of_kv_heads: int):
        super().__init__()
        self.model_dimension = model_dimension
        self.heads_dimension = model_dimension // number_of_heads
        self.number_of_heads = number_of_heads
        self.number_of_kv_heads = number_of_kv_heads
        self.repeats = self.number_of_heads // self.number_of_kv_heads

        self.q_projector = Linear(model_dimension, self.heads_dimension * self.number_of_heads, bias=False)
        self.k_projector = Linear(model_dimension, self.heads_dimension * self.number_of_kv_heads, bias=False)
        self.v_projector = Linear(model_dimension, self.heads_dimension * self.number_of_kv_heads, bias=False)
        self.output_projector = Linear(self.number_of_heads * self.heads_dimension, model_dimension, bias=False)
        
    def forward(self, sequence: Tensor, rotatory_embeddings: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        query, key, value = self.q_projector(sequence), self.k_projector(sequence), self.v_projector(sequence)        
        query, key, value = split_heads(query, self.number_of_heads), split_heads(key, self.number_of_kv_heads), split_heads(value, self.number_of_kv_heads)
        query, key = apply_rotatory_embeddings(query, rotatory_embeddings), apply_rotatory_embeddings(key, rotatory_embeddings)
        attention = scaled_dot_product_attention(query, key, value, mask)
        attention = concat_heads(attention)
        return self.output_projector(attention)