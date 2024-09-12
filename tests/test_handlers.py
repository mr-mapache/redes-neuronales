import pytest

from src.adapters.states import States
from src.adapters.experiments import Experiments, Settings
from src.application.commands import (
    Command, 
    CreateExperiment,
    TrainOverEpochs
)

from src.application.handlers import (
    handle_create_experiment,
    handle_training_over_epochs
)

States.register('mlp-cls-784-128-10-0.5-relu', 'classifier')
States.register('cross-entropy', 'criterion')
States.register('adam-0.001', 'optimizer')

@pytest.fixture
def settings(mongo_database, directory) -> Settings:
    return Settings(
        device='cpu',
        workers=None,
        database=mongo_database,
        directory=directory
    )

@pytest.fixture
def experiments(settings: Settings):
    return Experiments(settings)

@pytest.fixture
def create_experiment() -> Command:
    return CreateExperiment(
        name='test',
        nn='mlp-cls-784-128-10-0.5-relu',
        criterion='cross-entropy',
        optimizer='adam-0.001',
        batch_size=32
    )

@pytest.fixture
def train_over_epochs() -> Command:
    return TrainOverEpochs(
        experiment='test',
        dataset='mnist',
        task='classification',
        epochs=1
    )

def test_handle_create_experiment(create_experiment: CreateExperiment, experiments: Experiments):
    handle_create_experiment(create_experiment, experiments)
    experiment = experiments.get_by_name(create_experiment.name)
    assert experiment.name == create_experiment.name
    assert experiment.id is not None
    
    state = experiments.states.get(experiment)
    assert state.nn == create_experiment.nn
    assert state.criterion == create_experiment.criterion
    assert state.optimizer == create_experiment.optimizer

def test_handle_training_over_epochs(
    create_experiment: CreateExperiment, 
    train_over_epochs: TrainOverEpochs, 
    experiments: Experiments
):
    with pytest.raises(Exception):
        handle_training_over_epochs(train_over_epochs, experiments)

    handle_create_experiment(create_experiment, experiments)
    handle_training_over_epochs(train_over_epochs, experiments)

    experiment = experiments.get_by_name(train_over_epochs.experiment)
    state = experiments.states.get(experiment)
    assert state.epochs == 1
    metrics = experiments.metrics.pull(experiment)
    accuracy = metrics['accuracy']
    assert accuracy.name == 'accuracy'
    assert len(accuracy.history['train']) == 1
    assert len(accuracy.history['evaluation']) == 1
    loss = metrics['loss']
    assert loss.name == 'loss'
    assert len(loss.history['train']) == 1
    assert len(loss.history['evaluation']) == 1
    model = experiments.states.models.get(state, experiment)
    assert model is not None
    assert model.nn is not None
    assert model.criterion is not None
    assert model.optimizer is not None 