from logging import getLogger
from typing import Callable, Dict, Any, Union

from torch import compile
from torch.optim import Optimizer
from torch.nn import Module

from src.adapters.models.entity import Model

logger = getLogger(__name__)

def create_model(nn, criterion, optimizer, device) -> Model:
    if criterion and optimizer:
        model = Model(nn, criterion, optimizer)
        if device:
            model.to(device)
        try:
            logger.info(f'Compiling model...')
            model = compile(model)
            return model
        except Exception as exception:
            logger.exception('Failed to compile model, returning uncompiled model')
            logger.exception(exception)            
            return model
    else:
        raise NotImplementedError('Inference mode not implemented yet')

class Builder:
    nn_factory: Dict[str, Callable[[], Module]] = {}
    criterion_factory: Dict[str, Callable[[], Callable]] = {}
    optimizer_factory: Dict[str, Callable[[Module], Optimizer]] = {}

    def __init__(self):
        self.name = None
        self.nn: Dict[str, Union[str, Module]] = {'name': None, 'nn': None}
        self.criterion: Dict[str, Union[str, Callable]] = {'name': None, 'criterion': None}
        self.optimizer: Dict[str, Union[str, Optimizer]] = {'name': None, 'optimizer': None}
        self.device = 'cpu'

    @classmethod
    def register_nn(cls, nn_name: str, nn_cls: Callable[[], Module]):
        cls.nn_factory[nn_name] = nn_cls

    @classmethod
    def register_criterion(cls, criterion_name: str, criterion_cls: Callable[[], Callable]):
        cls.criterion_factory[criterion_name] = criterion_cls

    @classmethod
    def register_optimizer(cls, optimizer_name: str, optimizer_cls: Callable[[Module], Optimizer]):
        cls.optimizer_factory[optimizer_name] = optimizer_cls

    def set_nn(self, nn_name: str):
        if nn_name not in self.nn_factory:
            raise ValueError(f'Unknown neural network: {nn_name}')
        self.nn['name'] = nn_name
        self.nn['nn'] = self.nn_factory[nn_name]()
    
    def set_criterion(self, criterion_name: str):
        if criterion_name not in self.criterion_factory:
            raise ValueError(f'Unknown criterion: {criterion_name}')
        self.criterion['name'] = criterion_name
        self.criterion['criterion'] = self.criterion_factory[criterion_name]()
    
    def set_optimizer(self, optimizer_name: str):
        if not self.nn['nn']:
            raise ValueError("Model must be set before setting the optimizer.")
        if optimizer_name not in self.optimizer_factory:
            raise ValueError(f'Unknown optimizer: {optimizer_name}')
        self.optimizer['name'] = optimizer_name
        self.optimizer['optimizer'] = self.optimizer_factory[optimizer_name](self.nn['nn'])

    def set_device(self, device: str):
        self.device = device

    def build(self) -> Any:
        if not all([self.nn['nn'], self.criterion['criterion'], self.optimizer['optimizer']]):
            raise ValueError("Neural network, criterion, and optimizer must all be set before building.")
        return create_model(self.nn['nn'], self.criterion['criterion'], self.optimizer['optimizer'], device=self.device)