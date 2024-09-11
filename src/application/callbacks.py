from typing import Protocol
from typing import Optional
from logging import getLogger
from src.domain.models import Result, Phase

from src.domain.models import Metric
from src.adapters.metrics.values import Loss, Accuracy
from src.adapters.metrics.values import factory

logger = getLogger(__name__)

class Callback(Protocol):
    def __call__(self, result: Result):
        ...

    @property
    def metrics(self) -> list[Metric]:
        ...

class Classification:
    def __init__(self, loss: Loss, accuracy: Accuracy):
        self.loss = loss
        self.accuracy = accuracy
        self.batch = 0
        self.epoch = 0

    @property
    def metrics(self) -> list[Metric]:
        return [self.loss, self.accuracy]

    def __call__(self, result: Result):
        if result.phase == Phase.BREAK:
            logger.info(f'--------------------------------------------------------------------------')
            logger.info(f'--- Results: ::: Average Loss: {self.loss.average:.4f} ::: Average accuracy: {self.accuracy.average:.4f} ---')
            logger.info(f'--------------------------------------------------------------------------')
        loss = self.loss(result.batch, result.loss, result.phase)
        accuracy = self.accuracy(result.batch, result.output, result.target, result.phase)
        if result.batch % 100 == 0 and result.phase != Phase.BREAK:
            logger.info(f' --- --- --- ::: Batch {result.batch} ::: Loss {loss:.4f} ::: Accuracy {accuracy:.4f}')


def get_callback(task: str, metrics: dict[str, Metric]) -> Optional[Callback]:
    
    match task:
        case 'classification':
            return Classification(
                accuracy=metrics.get('accuracy', factory('accuracy')),
                loss=metrics.get('loss', factory('loss'))
            )
                                     
        case _:
            return None