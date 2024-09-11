from uuid import UUID
from typing import override
from typing import Optional
from dataclasses import dataclass, field 

from pymongo.database import Database

from src.domain.models import Experiment
from src.domain.ports import Experiments as Repository
from src.adapters.loaders import Loaders
from src.adapters.metrics import Metrics
from src.adapters.states import States

@dataclass(kw_only=True)
class Settings:
    device: Optional[str] = field(default='cpu')
    workers: Optional[int] = field(default=0)
    database: Database
    directory: str

class Experiments(Repository):
    def __init__(self, settings: Settings):
        self.device = settings.device
        self.collection = settings.database['experiments']
        self.loaders = Loaders(settings.device, settings.workers)
        self.metrics = Metrics(settings.database)
        self.states = States(settings.database, settings.directory, self.device)

    @override
    def add(self, experiment: Experiment):
        print(experiment.name)
        document = {'_id': str(experiment.id), 'name': experiment.name}
        self.collection.insert_one(document)

    @override
    def remove(self, experiment: Experiment):
        filter = {'_id': str(experiment.id)}
        self.collection.delete_one(filter)
    
    @override
    def update(self, experiment: Experiment):
        filter = {'_id': str(experiment.id)}
        update = {'$set': {'name': experiment.name}}
        self.collection.update_one(filter, update)

    @override
    def get(self, id: UUID) -> Experiment:
        filter = {'_id': str(id)}
        result = self.collection.find_one(filter)
        if result is None:
            return None
        return Experiment(name=result['name'], id=UUID(result['_id']))
    
    @override
    def get_by_name(self, name: str) -> Experiment:
        filter = {'name': name}
        result = self.collection.find_one(filter)
        if result is None:
            return None
        return Experiment(name=result['name'], id=UUID(result['_id']))
        
    @override
    def list(self) -> list[Experiment]:
        result = self.collection.find()
        return [Experiment(name=document['name'], id=UUID(document['_id'])) for document in result]