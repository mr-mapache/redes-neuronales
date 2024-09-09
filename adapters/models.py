import os
from typing import override
from torch import save, load

from ops.ports import Models as DAO
from ops.models import Experiment, Model

from logging import getLogger

logger = getLogger(__name__)

#TODO: save model representation into mongodb.

class Models(DAO):
    def __init__(self, directory: str):
        self.directory = directory

    @override
    def save(self, model: Model, experiment: Experiment):
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        save({
            'network_state_dict': model.network.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'criterion_state_dict': model.criterion.state_dict(),
        }, f'{self.directory}/{experiment.name}-{experiment.id}.pt')

    @override
    def restore(self, model: Model, experiment: Experiment):
        file = f'{self.directory}/{experiment.name}-{experiment.id}.pt'
        if os.path.exists(file):
            checkpoint = load(file, weights_only=False)
            model.network.load_state_dict(checkpoint['network_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.criterion.load_state_dict(checkpoint['criterion_state_dict'])

    @override
    def remove(self, experiment: Experiment):
        os.remove( f'{self.directory}/{experiment.name}-{experiment.id}.pt')