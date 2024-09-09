from typing import Optional
from uuid import UUID
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ops.models import Experiment
from ops.models import Metric
from ops.models import Model
from ops.models import Loader
from ops.factory import Factory

class Experiments(ABC):

    @abstractmethod
    def add(self, experiment: Experiment): ... 

    @abstractmethod
    def remove(self, experiment: Experiment): ...

    @abstractmethod
    def update(self, experiment: Experiment): ...

    @abstractmethod
    def get(self, id: UUID) -> Optional[Experiment]: ...

    @abstractmethod
    def get_by_name(self, name: str) -> Optional[Experiment]: ...

    @abstractmethod
    def list(self) -> list[Experiment]: ...


class Metrics(ABC):

    @abstractmethod
    def push(self, metric: Metric, phase: str, experiment: Experiment): ...

    @abstractmethod
    def pull(self, phase: str, experiment: Experiment) -> list[Metric]: ...


class Models(ABC):

    @abstractmethod
    def save(self, model: Model, experiment: Experiment): ...

    @abstractmethod
    def restore(self, model: Model, experiment: Experiment): ...

    @abstractmethod
    def remove(self, experiment: Experiment): ...


class Data(ABC):

    @abstractmethod
    def get(self, dataset: str, train: bool, batch_size: int) -> Loader: ...


@dataclass
class Repository:
    experiments: Experiments
    metrics: Metrics
    models: Models
    data: Data
    factory: Factory