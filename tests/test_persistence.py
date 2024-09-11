import os
import shutil
import pytest
from logging import getLogger
from uuid import UUID

from src.adapters.experiments import Experiments, Settings
from src.adapters.metrics import Metrics, factory
from src.adapters.models import Models, create_model
from src.adapters.loaders import Loaders
from src.adapters.states import States
from src.domain.models import Experiment, Phase, Model, State
from src.adapters.models.factory import Builder

from torch import allclose
from torch.nn import Sequential, Linear, ReLU
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from torch.optim import Adam

from models.classifiers import MLP as MLPCLS


logger = getLogger(__name__)

@pytest.fixture
def settings(mongo_database, directory) -> Settings:
    return Settings(
        device='cpu',
        workers=None,
        database=mongo_database,
        directory=directory
    )

@pytest.fixture
def models(directory) -> Models:
    return Models(directory)

@pytest.fixture
def metrics(mongo_database):
    return Metrics(mongo_database)

@pytest.fixture
def model() -> Model:
    nn = Sequential(Linear(12, 6), ReLU(), Linear(6, 3))
    criterion = CrossEntropyLoss()
    optimizer = SGD(nn.parameters(), lr=0.01)
    return create_model(
        nn=nn,
        criterion=criterion,
        optimizer=optimizer,
        device='cpu'
    )


@pytest.fixture
def loaders() -> Loaders:
    return Loaders('cpu', None)

    
@pytest.fixture
def states(mongo_database, directory):
    States.register('mlp-cls-784-128-10-0.5-relu', 'classifier')
    States.register('cross-entropy', 'criterion')
    States.register('adam-0.001', 'optimizer')
    return States(mongo_database, directory)

@pytest.fixture
def experiments(settings: Settings):
    return Experiments(settings)

def test_experiments(experiments: Experiments):
    experiment = Experiment(id=UUID('00000000-0000-0000-0000-000000000000'), name='test')
    experiments.add(experiment)
    assert experiments.get_by_name('test') == experiment
    result = experiments.get('00000000-0000-0000-0000-000000000000')
    assert result == experiment
    assert result.name == 'test'

    experiment.name = 'new name'
    experiments.update(experiment)
    
    result = experiments.get('00000000-0000-0000-0000-000000000000')
    assert result.name == 'new name'

    assert experiments.list() == [experiment]
    experiments.remove(experiment)
    assert experiments.get_by_name('test') is None
    assert experiments.list() == []
    logger.info('All tests passed')       

def test_metrics(metrics: Metrics):
    experiment = Experiment(id=UUID('00000000-0000-0000-0000-000000000000'), name='test')
    accuracy = factory('accuracy', {
        Phase.TRAIN: [0.1, 0.2],
        Phase.EVALUATION: [0.3, 0.4]
    })
    metrics.push(accuracy, experiment)
    metrics.push(accuracy, experiment)
    loss = factory('loss', {
        Phase.TRAIN: [0.5, 0.6],
        Phase.EVALUATION: [0.7, 0.8]
    })
    metrics.push(loss, experiment)
    metrics_dict = metrics.pull(experiment)
    
    assert metrics_dict['accuracy'].name == 'accuracy'
    assert metrics_dict['accuracy'].history[Phase.TRAIN] == [0.1, 0.2]
    assert metrics_dict['accuracy'].history[Phase.EVALUATION] == [0.3, 0.4]
    assert metrics_dict['loss'].name == 'loss'

    accuracy.history[Phase.TRAIN].extend([0.9, 1.0])
    metrics.push(accuracy, experiment)
    metrics_dict = metrics.pull(experiment)
    assert metrics_dict['accuracy'].history[Phase.TRAIN] == [0.1, 0.2, 0.9, 1.0]
    assert metrics_dict['accuracy'].history[Phase.EVALUATION] == [0.3, 0.4]
    assert metrics_dict['loss'].history[Phase.TRAIN] == [0.5, 0.6]


def test_model(models: Models):
    experiment = Experiment(id=UUID('00000000-0000-0000-0000-000000000000'), name='test')
    nn = Sequential(Linear(12, 6), ReLU(), Linear(6, 3))
    criterion = CrossEntropyLoss()
    optimizer = SGD(nn.parameters(), lr=0.01)
    model = create_model(
        nn=nn,
        criterion=criterion,
        optimizer=optimizer,
        device='cpu'
    )

    models.save(model, experiment)
    other_network = Sequential(Linear(12, 6), ReLU(), Linear(6, 3))
    other_criterion = CrossEntropyLoss()
    other_optimizer = SGD(other_network.parameters(), lr=0.02)
    other_model = create_model(nn=other_network, criterion=other_criterion, optimizer=other_optimizer, device='cpu')
    models.restore(other_model, experiment)
    assert allclose(other_model.nn[0].weight, model.nn[0].weight)
    assert other_model.nn[1].__class__ == model.nn[1].__class__
    assert other_model.criterion.__class__ == model.criterion.__class__
    assert other_model.optimizer.param_groups[0]['lr'] == 0.01  


def test_states(states: States):

    experiment = Experiment(id=UUID('00000000-0000-0000-0000-000000000000'), name='test')
    state = State(
        nn='mlp-cls-784-128-10-0.5-relu',
        criterion='cross-entropy',
        optimizer='adam-0.001',
        batch_size=32,
        epochs=0
    )

    states.add(state, experiment)
    result = states.get(experiment)
    assert result.nn == 'mlp-cls-784-128-10-0.5-relu'
    assert result.criterion == 'cross-entropy'
    assert result.optimizer == 'adam-0.001'
    assert result.batch_size == 32
    assert result.epochs == 0

    state.epochs = 10

    states.update(state, experiment)
    result = states.get(experiment)
    assert result.epochs == 10

    states.remove(experiment)
    assert states.get(experiment) is None

    assert states.verify(state) == True
    state.nn = 'unknown'
    assert states.verify(state) == False

    
def test_datasets(loaders: Loaders):
    loader = loaders.get('mnist', train=True, batch_size=32)
    for input, target in loader:
        assert input.shape == (32, 1, 28, 28)
        assert target.shape == (32,)
        break

def test_models_consistency(models: Models, experiments: Experiments, states: States):
    Builder.register_nn('mlp-cls-784-128-10-0.5-relu', lambda: MLPCLS(784, 128, 10, 0.5))
    Builder.register_criterion('cross-entropy', CrossEntropyLoss)
    Builder.register_optimizer('adam-0.001', lambda model: Adam(model.parameters(), lr=0.001))

    experiment = Experiment(id=UUID('00000000-0000-0000-0000-000000000000'), name='test')
    state = State(
        nn='mlp-cls-784-128-10-0.5-relu',
        criterion='cross-entropy',
        optimizer='adam-0.001',
        batch_size=32,
        epochs=3
    )

    builder = Builder()
    builder.set_device('cpu')
    builder.set_nn(state.nn)
    builder.set_optimizer(state.optimizer)
    builder.set_criterion(state.criterion)
    model = builder.build()
    
    experiments.add(experiment)
    states.add(state, experiment)
    models.save(model, experiment)

    restored_model = models.get(state, experiment)
    assert restored_model is not None
    
    assert allclose(restored_model.nn.input_layer.weight, model.nn.input_layer.weight)
    assert restored_model.optimizer.param_groups[0]['lr'] == model.optimizer.param_groups[0]['lr']