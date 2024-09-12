from typing import Optional
from typing import Callable
from uuid import UUID
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.domain.models import Experiment
from src.domain.models import Model, Metric
from src.domain.models import Loader
from src.domain.models import State

class Models(ABC):

    @abstractmethod
    def save(self, model: Model, experiment: Experiment): ...

    @abstractmethod
    def restore(self, model: Model, experiment: Experiment): ...

    @abstractmethod
    def remove(self, experiment: Experiment): ...

    @abstractmethod
    def get(self, state: State, experiment: Experiment) -> Model: ...
    

class Metrics(ABC):
    register: Callable
    
    @abstractmethod
    def create(self, name: str, history: dict | None = None) -> Metric: ...

    @abstractmethod
    def push(self, metric: Metric, experiment: Experiment): ...

    @abstractmethod
    def pull(self, experiment: Experiment) -> dict[str, Metric]: ...

    @abstractmethod
    def get(self, name: str, experiment: Experiment) -> Optional[Metric]: ...


class States(ABC):
    models: Models

    @abstractmethod
    def add(self, state: State, experiment: Experiment): ...

    @abstractmethod
    def update(self, state: State, experiment: Experiment): ...

    @abstractmethod
    def get(self, experiment: Experiment) -> State: ...

    @abstractmethod
    def remove(self, experiment: Experiment): ...

    @abstractmethod
    def verify(self, state: State) -> bool: ...


class Loaders(ABC):

    @abstractmethod
    def get(self, dataset: str, batch_size: int, train: bool) -> Loader: ...

class Experiments(ABC):
    device: str
    metrics: Metrics
    states: States
    loaders: Loaders

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