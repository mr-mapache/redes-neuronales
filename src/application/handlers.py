from uuid import uuid3, NAMESPACE_DNS
from time import time
from typing import Literal
from dataclasses import dataclass, field
from src.domain.models import Experiment, State
from src.domain.ports import Experiments
from src.domain.services import train, evaluate

from src.application import exceptions
from src.application.callbacks import get_callback
from src.application.commands import (
    CreateExperiment,
    TrainOverEpochs
)

from logging import getLogger

logger = getLogger(__name__)

def handle_create_experiment(command: CreateExperiment, experiments: Experiments):
    experiment = experiments.get_by_name(command.name)
    if experiment:
        raise exceptions.ExperimentAlreadyExists(f'Experiment {command.name} already exists')
    experiment = Experiment(id=uuid3(NAMESPACE_DNS, command.name), name=command.name)   
    state = State(nn=command.nn, criterion=command.criterion, optimizer=command.optimizer,batch_size=command.batch_size, epochs=0)
    if not experiments.states.verify(state):
        raise exceptions.ModelNotSupported(f'Model not supported')
    experiments.add(experiment)
    experiments.states.add(state, experiment)


def handle_training_over_epochs(command: TrainOverEpochs, experiments: Experiments):
    logger.info(f'Recovering experiment {command.experiment}...')

    experiment = experiments.get_by_name(command.experiment)
    if not experiment:
        raise exceptions.ExperimentNotFound(f'Experiment {command.experiment} not found')
    
    state = experiments.states.get(experiment)
    if not state:
        raise exceptions.StateNotFound(f'State for experiment {command.experiment} not found')
    
    model = experiments.states.models.get(state, experiment)
    metrics = experiments.metrics.pull(experiment)

    logger.info(f'Experiment {command.experiment} recovered successfully')
    logger.info(f'Starting training from epoch {state.epochs}')
    
    train_loader = experiments.loaders.get(command.dataset, True, state.batch_size, transform=command.transform)
    test_loader = experiments.loaders.get(command.dataset, False, state.batch_size, transform=command.transform)

    callback = get_callback(command.task, metrics)
    if not callback:
        raise exceptions.TaskNotSupported(f'Task {command.task} not supported')

    try:
        start = time()

        for epoch in range(command.epochs):
            logger.info(f' --- Starting epoch {epoch+1} ... ---')
            train(model, train_loader, callback, experiments.device)
            evaluate(model, test_loader, callback, experiments.device)


        end = time()
        logger.info(f'--- End of training ---')
        logger.info(f'Training took {end-start} seconds over {command.epochs} epochs')
        logger.info(f'Saving experiment {command.experiment}...')           
        
        state.epochs += command.epochs
        experiments.states.update(state, experiment)
        for metric in callback.metrics:
            experiments.metrics.push(metric, experiment)
        experiments.states.models.save(model, experiment)

        logger.info(f'Experiment {command.experiment} saved successfully')
        logger.info(f'Total epochs of the experiment: {state.epochs}')

    except KeyboardInterrupt:
        return