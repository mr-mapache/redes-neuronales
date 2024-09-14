from typing import Tuple
from torch import Tensor
from torch import cat, randn, flatten
from torch.nn import Parameter
from torch.nn import Module
from torch.nn import Conv2d

def number_of_patches(image_shape: Tuple[int, int], patch_shape: Tuple[int, int]) -> int:
    image_height, image_width = image_shape
    patch_height, patch_width = patch_shape
    assert image_height % patch_height == 0 and image_width % patch_width == 0, 'image shape must be divisible by patch shape'
    return (image_height // patch_height) * (image_width // patch_width)

class ConvolutionalPatch(Module):
    def __init__(self, model_dimension: int, patch_shape: Tuple[int, int], number_of_channels: int):
        super().__init__()
        self.projector = Conv2d(number_of_channels, model_dimension, kernel_size=patch_shape, stride=patch_shape)

    def forward(self, input: Tensor) -> Tensor:
        output = self.projector(input)
        return flatten(output, 2).transpose(1, 2)    

class CLSToken(Module):
    def __init__(self, model_dimension: int):
        super().__init__()
        self.token = Parameter(randn(1, 1, model_dimension))

    def forward(self, input: Tensor) -> Tensor:
        batch_size = input.shape[0]
        token = self.token.expand(batch_size, -1, -1)
        return cat([token, input], dim=1)

class LearnablePositionalEncoding(Module):
    def __init__(self, model_dimension: int, sequence_lenght_limit: int = 196):
        super().__init__()
        self.sequence_lenght_limit = sequence_lenght_limit
        self.position_embeddings = Parameter(randn(1, sequence_lenght_limit + 1, model_dimension))

    def forward(self, input: Tensor) -> Tensor:
        assert input.size(1) <= self.sequence_lenght_limit + 1, 'input sequence is too long'
        input = input + self.position_embeddings[:, :input.size(1)]
        return input