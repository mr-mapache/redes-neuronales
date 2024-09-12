from typing import Literal
from pydantic import BaseModel
from pydantic import Field

class Command(BaseModel):
    pass

class CreateExperiment(Command):
    name: str = Field(...)
    nn: str = Field(...)
    criterion: str = Field(...)
    optimizer: str = Field(...)
    batch_size: int = Field(...)

class TrainOverEpochs(Command):
    epochs: int = Field(...)
    experiment: str = Field(...)
    dataset: str = Field(...)
    task: Literal['classification'] = Field(default='classification')