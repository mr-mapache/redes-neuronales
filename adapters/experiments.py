from uuid import UUID
from typing import override
from pymongo.database import Database

from ops.models import Experiment
from ops.ports import Experiments as DAO
from ops.factory import Factory

class Experiments(DAO):
    def __init__(self, database: Database, factory: Factory):
        self.collection = database['experiments']
        self.factory = factory

    @override
    def add(self, experiment: Experiment):
        document = {'_id': str(experiment.id), 'name': experiment.name, 'epochs': experiment.epochs}
        self.collection.insert_one(document)

    @override
    def remove(self, experiment: Experiment):
        filter = {'_id': str(experiment.id)}
        self.collection.delete_one(filter)
    
    @override
    def update(self, experiment: Experiment):
        filter = {'_id': str(experiment.id)}
        self.collection.update_one(filter, {'name': experiment.name, 'epochs': experiment.epochs})

    @override
    def get(self, id: UUID) -> Experiment:
        filter = {'_id': str(id)}
        result = self.collection.find_one(filter)
        if result is None:
            return None
        return self.factory.experiment(name=result['name'], id=result['_id'], epochs=result['epochs'])
    
    @override
    def get_by_name(self, name: str) -> Experiment:
        filter = {'name': name}
        result = self.collection.find_one(filter)
        if result is None:
            return None
        return self.factory.experiment(name=result['name'], id=result['_id'], epochs=result['epochs'])
        
    @override
    def list(self) -> list[Experiment]:
        result = self.collection.find()
        return [self.factory.experiment(name=document['name'], id=document['_id'], epochs=document['epochs']) for document in result]