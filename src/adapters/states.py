from uuid import UUID
from typing import override
from typing import Optional
from pymongo.database import Database

from src.domain.models import Experiment, State
from src.domain.ports import States as Repository
from src.adapters.models import Models, create_model

class States(Repository):
    registry : dict[str, str] = {}

    @classmethod
    def register(cls, name: str, kind: str):
        cls.registry[name] = kind

    def __init__(self, database: Database, directory: str, device: str = 'cpu'):
        self.collection = database['states']
        self.models = Models(directory, device = device)
        self.registry_keys = self.registry.keys()

    @override
    def add(self, state: State, experiment: Experiment):
        document = {'_id': str(experiment.id), 'state': {
            'nn': state.nn,
            'criterion': state.criterion,
            'optimizer': state.optimizer,
            'batch_size': state.batch_size,
            'epochs': state.epochs
        }}
        self.collection.insert_one(document)

    @override
    def update(self, state: State, experiment: Experiment):
        filter = {'_id': str(experiment.id)}
        update = {'$set': {'state': {
            'nn': state.nn,
            'criterion': state.criterion,
            'optimizer': state.optimizer,
            'batch_size': state.batch_size,
            'epochs': state.epochs
        }}}
        self.collection.update_one(filter, update)

    @override
    def get(self, experiment: Experiment) -> Optional[State]:
        filter = {'_id': str(experiment.id)}
        result = self.collection.find_one(filter)
        if result is None:
            return None
        return State(
            nn=result['state']['nn'],
            criterion=result['state']['criterion'],
            optimizer=result['state']['optimizer'],
            batch_size=result['state']['batch_size'],
            epochs=result['state']['epochs']
        )
    
    @override
    def remove(self, experiment: Experiment):
        filter = {'_id': str(experiment.id)}
        self.collection.delete_one(filter)

    @override
    def verify(self, state: State) -> bool:
        valid_nn = True if state.nn in self.registry_keys else False
        valid_criterion = True if state.criterion in self.registry_keys else False
        valid_optimizer = True if state.optimizer in self.registry_keys else False
        return valid_nn and valid_criterion and valid_optimizer