from typing import Literal, Callable
from torch.nn import Module
from torch.optim import Optimizer
from src.adapters.models.factory import Builder
from src.adapters.states import States
from src.adapters.experiments import Experiments

def register(part: Literal['nn', 'criterion', 'optimizer'], name: str, factory: Callable[[], Callable[[None], Module | Optimizer]]):
    match part:
        case 'nn':
            Builder.register_nn(name, factory)
            States.register(name, 'nn')
        case 'criterion':
            Builder.register_criterion(name, factory)
            States.register(name, 'criterion')  
        case 'optimizer':
            Builder.register_optimizer(name, factory)
            States.register(name, 'optimizer')
        case _:
            raise ValueError(f'Unknown model part: {part}')