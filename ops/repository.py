import os
from typing import override
from typing import Optional
from abc import ABC, abstractmethod
from datetime import datetime
from logging import getLogger

from ops.model import Model
from ops.ports import Publisher, Repository

from torch import save, load, compile
from torch.nn import Module
from torch.optim import Optimizer

class Factory:
    def __init__(self, publisher: Publisher):
        self.publisher = publisher
    
    def create(self, name: str, network: Module, criterion: Module, optimizer: Optimizer, device: str) -> Model:
        logger = getLogger(__name__)
        model = Model(name, network, criterion, optimizer).to(device)
        try:
            logger.info(f'Compiling model {name}')
            model = compile(model)
            return model
        except Exception as exception:
            logger.exception('Failed to compile model, returning uncompiled model')
            logger.exception(exception)            
            return model


class Models(Repository):
    def __init__(self, path: str, device: str, publisher: Publisher):
        self.path = path
        self.device = device
        self.factory = Factory(publisher)

    @override
    def create(self, name: str, network: Module, criterion: Module, optimizer: Optimizer) -> Model:
        name += '-' + network.__class__.__name__
        name += '-' + criterion.__class__.__name__
        name += '-' + optimizer.__class__.__name__
        return self.factory.create(name, network, criterion, optimizer, self.device)
    
    @override
    def save(self, model: Model):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        file = f'{self.path}/{model.name}.pt'
        save({
            'network_state_dict': model.network.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'criterion_state_dict': model.criterion.state_dict(),
            'epochs': model.epochs
        }, file)
        
    @override
    def restore(self, model: Model):
        file = f'{self.path}/{model.name}.pt'
        if os.path.exists(file):
            checkpoint = load(file, weights_only=False)
            model.network.load_state_dict(checkpoint['network_state_dict'])
            model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            model.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            model.epochs = checkpoint['epochs']            

    @override
    def delete(self, name: str):
        path = f'{self.path}/{name}.pt'
        os.remove(path)