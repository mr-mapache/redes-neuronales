from uuid import UUID
from logging import getLogger
from typing import Generator

from fastapi import FastAPI
from fastapi import HTTPException, Depends, status
from fastapi import Request
from fastapi import Query
from fastapi.responses import Response

from src.domain.models import Experiment, Metric
from src.domain.ports import Experiments
from src.application.commands import CreateExperiment, TrainOverEpochs
from src.application.exceptions import ExperimentAlreadyExists, ExperimentNotFound, ModelNotSupported
from src.application.handlers import handle_create_experiment, handle_training_over_epochs
from src.application.messagebus import Messagebus

api = FastAPI()
logger = getLogger(__name__)

def repository() -> Experiments:
    raise NotImplementedError

def bus() -> Messagebus:
    raise NotImplementedError

@api.post('/experiments/')
def create_experiment(command: CreateExperiment, experiments: Experiments = Depends(repository)):
    experiment = handle_create_experiment(command, experiments)
    return experiment

@api.exception_handler(ExperimentAlreadyExists)
def handle_experiment_already_exists(request: Request, exception: Exception):
    logger.error(exception)
    logger.error(f'Error on {request.url.path}')
    raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exception))

@api.exception_handler(ExperimentNotFound)
def handle_experiment_not_found(request: Request, exception: Exception):
    logger.error(exception)
    logger.error(f'Error on {request.url.path}')
    raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exception))

@api.exception_handler(ModelNotSupported)
def handle_model_not_supported(request: Request, exception: Exception):
    logger.error(exception)
    logger.error(f'Error on {request.url.path}')
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exception))

@api.get('/experiments/')
def list_experiments(experiments: Experiments = Depends(repository)) -> list[Experiment]:
    return experiments.list()

@api.get('/experiments/name')
def get_experiment_by_name(name: str = Query(...), experiments: Experiments = Depends(repository)) -> Experiment:
    experiment = experiments.get_by_name(name)
    if not experiment:
        raise ExperimentNotFound(f'Experiment {name} not found')
    return experiment

@api.get('/experiments/{id}/metrics/')
def get_experiment_metrics(id: UUID, experiments: Experiments = Depends(repository)) -> dict[str, Metric]:
    experiment = experiments.get(id)
    if not experiment:
        raise ExperimentNotFound(f'Experiment {id} not found')
    metrics = experiments.metrics.pull(experiment)
    return metrics

@api.post('/experiments/train/')
def train_over_epochs(command: TrainOverEpochs, experiments: Experiments = Depends(repository)):
    handle_training_over_epochs(command, experiments)
    return Response(status_code=status.HTTP_202_ACCEPTED)