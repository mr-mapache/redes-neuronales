from os import remove
from os import makedirs
from os import path
from typing import override
from typing import Optional, Callable
from logging import getLogger

from torch import save, load
from pymongo.database import Database

from src.domain.models import Experiment
from src.domain.ports import Models as Repository
from src.adapters.models.entity import Model
from src.adapters.models.factory import create_model, Builder
from src.domain.models import State

logger = getLogger(__name__)

class Models(Repository):
    def __init__(self, directory: str, factory: Callable = create_model, device: str = 'cpu'):
        self.directory = directory
        self.factory = factory
        self.device = device

    def path(self, experiment: Experiment) -> str:
        return f'{self.directory}/{experiment.name}-{experiment.id}.pt'

    @override
    def save(self, model: Model, experiment: Experiment):
        if not path.exists(self.directory):
            makedirs(self.directory)        
        if model.optimizer and model.criterion:
            save({
                'nn_state_dict': model.nn.state_dict(),
                'optimizer_state_dict': model.optimizer.state_dict(),
                'criterion_state_dict': model.criterion.state_dict()
            }, self.path(experiment))
        else:
            save({'nn_state_dict':model.nn.state_dict()}, self.path(experiment))

    @override
    def restore(self, model: Model, experiment: Experiment) -> Optional[Model]:
        if path.exists(self.path(experiment)):
            checkpoint = load(self.path(experiment), weights_only=False)
            model.nn.load_state_dict(checkpoint['nn_state_dict'])
            if model.optimizer and model.criterion:
                model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                model.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            return model
        else:
            logger.error(f'Model not found for experiment {experiment.id}')
            return None
        
    @override
    def remove(self, experiment: Experiment):
        remove( f'{self.directory}/{experiment.name}-{experiment.id}.pt')

    @override
    def get(self, state: State, experiment: Experiment) -> Model:
        builder = Builder()
        builder.set_device(self.device)
        builder.set_nn(state.nn)
        builder.set_optimizer(state.optimizer)
        builder.set_criterion(state.criterion)
        model = builder.build()
        return self.restore(model, experiment) or model