from typing import override
from typing import Callable
from typing import Optional
from logging import getLogger

from pymongo.database import Database

from src.domain.models import Experiment, Metric, Phase
from src.domain.ports import Metrics as Repository
from src.adapters.metrics.values import factory

logger = getLogger(__name__)

class Metrics(Repository):
    def __init__(self, database: Database, factory : Callable = factory):
        self.collection = database['metrics']
        self.factory = factory

    @override
    def create(self, name: str, history: dict | None = None) -> Metric:
        return self.factory(name, history)

    @override
    def push(self, metric: Metric, experiment: Experiment):
        filter = {'_id': str(experiment.id)}
        update = {
            '$set': {
                f'metrics.{metric.name}': {
                    'train': metric.history[Phase.TRAIN],
                    'evaluation': metric.history[Phase.EVALUATION]
                }
            }
        }
        self.collection.update_one(filter, update, upsert=True)
        
    @override
    def pull(self, experiment: Experiment) -> dict[str, Metric]:
        filter = {'_id': str(experiment.id)}
        document = self.collection.find_one(filter)
        if document is None:
            return {}
        return {name: self.factory(name, document['metrics'][name]) for name in document['metrics']}
    
    @override
    def get(self, name: str, experiment: Experiment) -> Optional[Metric]:
        filter = {'_id': str(experiment.id)}
        document = self.collection.find_one(filter)
        if document is None or name not in document['metrics']:
            return None
        return self.factory(name, document['metrics'][name])