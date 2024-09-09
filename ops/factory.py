from uuid import uuid4, UUID
from ops.models import Experiment
from ops.models import Metric
from ops.models import Model

from torch import compile
from torch.nn import Module
from torch.optim import Optimizer
from logging import getLogger

logger = getLogger(__name__)

def build(network: Module, criterion: Module, optimizer: Optimizer, device: str) -> Model:
    model = Model(network, criterion, optimizer).to(device)
    try:
        logger.info(f'Compiling model')
        model = compile(model)
        return model
    except Exception as exception:
        logger.exception('Failed to compile model, returning uncompiled model')
        logger.exception(exception)            
        return model

class Factory:
    def __init__(self):
        self.builders = {
            'experiment': lambda name, id: Experiment(id, name),
            'metric': lambda name, history: Metric(name, history),
            'model': build
        }	

    def experiment(self, name: str, id: UUID | str | None = None, epochs: int = 0) -> Experiment:
        if isinstance(id, str):
            id = UUID(id)
        experiment: Experiment = self.builders['experiment'](name, id or uuid4())
        experiment.epochs = epochs
        return experiment
    
    def metric(self, name: str, history: list[float] | None = None) -> Metric:
        return self.builders['metric'](name, history or [])
    
    def model(self, network: Module, criterion: Module, optimizer: Optimizer, device: str) -> Model:
        return self.builders['model'](network, criterion, optimizer, device)