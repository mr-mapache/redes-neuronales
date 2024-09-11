from typing import Literal
from dataclasses import dataclass, field

class Command:
    pass

@dataclass(kw_only=True)
class TrainOverEpochs(Command):
    epochs: int
    experiment: str
    dataset: str
    transform: str | None = field(default=None)
    task: Literal['classification'] = field(default='classification')

@dataclass(kw_only=True)
class CreateExperiment(Command):
    name: str
    nn: str
    criterion: str
    optimizer: str
    batch_size: int