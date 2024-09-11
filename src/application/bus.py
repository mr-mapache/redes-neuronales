from typing import Callable
from typing import Literal
from src.adapters.experiments import Experiments
from src.adapters.models.factory import Builder
from src.adapters.states import States
from src.domain.ports import Experiments
from src.application.commands import Command
from src.application.commands import CreateExperiment, TrainOverEpochs
from src.application.handlers import handle_create_experiment, handle_training_over_epochs
from src.application.exceptions import ExperimentAlreadyExists
from torch.nn import Module
from torch.optim import Optimizer
from logging import getLogger

logger = getLogger(__name__)


def handle_exception(exception: Exception):
    if isinstance(exception, ExperimentAlreadyExists):
        logger.exception('Experiment already exists')
        logger.info('Moving on...')
    else:
        raise exception

class MessageBus:   

    @classmethod
    def register(cls, part: Literal['nn', 'criterion', 'optimizer'], name: str, factory: Callable[[], Callable[[None], Module | Optimizer]]):
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
            

    def __init__(self, experiments: Experiments):
        self.experiments = experiments
        self.handlers: dict[Command, Callable[[Command], None]] = {
            CreateExperiment: lambda command: handle_create_experiment(command, self.experiments),
            TrainOverEpochs: lambda command: handle_training_over_epochs(command, self.experiments)
        }
        
    def handle(self, command: Command):
        handler = self.handlers.get(type(command), None)
        try:
            handler(command)
        except Exception as exception:
            handle_exception(exception)
            

