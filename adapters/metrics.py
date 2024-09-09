from uuid import UUID
from typing import override

from pymongo.database import Database
from ops.models import Experiment, Metric
from ops.ports import Metrics as DAO
from ops.factory import Factory

from logging import getLogger
logger = getLogger(__name__)

class Metrics(DAO):
    def __init__(self, database: Database, factory : Factory):
        self.collection = database['metrics']
        self.factory = factory

    @override
    def push(self, metric: Metric, phase: str, experiment: Experiment):
        filter = {'_id': str(experiment.id)}
        update = {
            '$push': {
                f'metrics.{phase}.{metric.name}': {
                    '$each': metric.history
                }
            }
        }

        self.collection.update_one(filter, update, upsert=True)
        
    @override
    def pull(self, phase: str, experiment: Experiment) -> list[Metric]:
        filter = {'_id': str(experiment.id)}
        projection = {f'metrics.{phase}': 1, '_id': 0}
        document = self.collection.find_one(filter, projection)        
        metrics: dict = document['metrics'][phase]
        return [self.factory.metric(name=name, history=history) for name, history in metrics.items()]