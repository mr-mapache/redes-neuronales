from typing import Optional
from torch import Tensor
from torch import repeat_interleave
from torch.nn import Module
from torch.nn import Linear
from torch.nn.functional import scaled_dot_product_attention
from models.transformers.ffn import FFN
from models.transformers.encoding import Rotatory, Learnable, LearnableRotatory

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
        key, value = repeat_interleave(key, self.repeats, 1), repeat_interleave(value, self.repeats, 1)
        attention = scaled_dot_product_attention(query, key, value, mask)
        attention = concat_heads(attention)
        return self.output_projector(attention)

class RopeMultiheadAttention(Attention):
    def __init__(self, model_dimension: int, number_of_heads: int, number_of_kv_heads: int, sequence_lenght_limit: int):
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
        self.positional_encoding = Rotatory(model_dimension, sequence_lenght_limit)
        
    def forward(self, sequence: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        query, key, value = self.q_projector(sequence), self.k_projector(sequence), self.v_projector(sequence)        
        query, key, value = split_heads(query, self.number_of_heads), split_heads(key, self.number_of_kv_heads), split_heads(value, self.number_of_kv_heads)
        query, key = self.positional_encoding(query), self.positional_encoding(key)
        key, value = repeat_interleave(key, self.repeats, 1), repeat_interleave(value, self.repeats, 1)
        attention = scaled_dot_product_attention(query, key, value, mask)
        attention = concat_heads(attention)
        return self.output_projector(attention)

class Encoder(Module):
    def __init__(self, attention_norm: Module, attention: Attention, ffn_norm: Module, ffn: FFN):
        super().__init__()
        self.attention_norm = attention_norm
        self.attention = attention
        self.ffn_norm = ffn_norm
        self.ffn = ffn
        
    def forward(self, input: Tensor) -> Tensor:
        output = input + self.attention(self.attention_norm(input))
        return output + self.ffn(self.ffn_norm(output))