import os
import pytest
from logging import getLogger
from uuid import UUID

from adapters.experiments import Experiments
from adapters.metrics import Metrics
from adapters.models import Models
from adapters.datasets import Data
from ops.models import Experiment, Metric, Model
from ops.factory import Factory

from torch import allclose
from torch.nn import Sequential, Linear, ReLU
from torch.nn import CrossEntropyLoss
from torch.optim import SGD

logger = getLogger(__name__)

@pytest.fixture
def factory():
    return Factory()

@pytest.fixture
def experiments(mongo_database, factory):
    return Experiments(mongo_database, factory)

def test_experiments(experiments: Experiments):
    experiment = Experiment(id=UUID('00000000-0000-0000-0000-000000000000'), name='test')
    experiments.add(experiment)
    assert experiments.get_by_name('test') == experiment
    result = experiments.get('00000000-0000-0000-0000-000000000000')
    assert result == experiment
    assert result.name == 'test'
    assert experiments.list() == [experiment]
    experiments.remove(experiment)
    assert experiments.get_by_name('test') is None
    assert experiments.list() == []
    logger.info('All tests passed')       
    
@pytest.fixture
def metrics(mongo_database, factory):
    return Metrics(mongo_database, factory)

def test_metrics(metrics: Metrics):
    experiment = Experiment(id=UUID('00000000-0000-0000-0000-000000000000'), name='test')

    metric = Metric(name='accuracy', history=[1., 2., 3.])
    metrics.push(metric, 'train', experiment)
    
    metric = Metric(name='accuracy', history=[3., 4., 5.])
    metrics.push(metric, 'train', experiment)

    metric = Metric(name='loss', history=[3., 4., 5.])
    metrics.push(metric, 'train', experiment)

    metric = Metric(name='loss', history=[1., 2., 3.])
    metrics.push(metric, 'test', experiment)

    metrics_list = metrics.pull('train', experiment)
    assert len(metrics_list) == 2
    assert metrics_list[0].name == 'accuracy'
    assert metrics_list[0].history == [1., 2., 3., 3., 4., 5.]
    assert metrics_list[1].name == 'loss'
    assert metrics_list[1].history == [3., 4., 5.]

    metrics_list = metrics.pull('test', experiment)
    assert len(metrics_list) == 1
    assert metrics_list[0].name == 'loss'
    assert metrics_list[0].history == [1., 2., 3.]

@pytest.fixture
def model(factory: Factory) -> Model:
    network = Sequential(Linear(12, 6), ReLU(), Linear(6, 3))
    criterion = CrossEntropyLoss()
    optimizer = SGD(network.parameters(), lr=0.01)
    return factory.model(
        network=network,
        criterion=criterion,
        optimizer=optimizer,
        device='cpu'
    )

@pytest.fixture
def models():
    yield Models('test')
    os.remove('test/test-00000000-0000-0000-0000-000000000000.pt')
    os.rmdir('test')

def test_model(models: Models, factory: Factory):
    experiment = Experiment(id=UUID('00000000-0000-0000-0000-000000000000'), name='test')
    network = Sequential(Linear(12, 6), ReLU(), Linear(6, 3))
    criterion = CrossEntropyLoss()
    optimizer = SGD(network.parameters(), lr=0.01)
    model = factory.model(network=network, criterion=criterion, optimizer=optimizer, device='cpu')

    models.save(model, experiment)
    other_network = Sequential(Linear(12, 6), ReLU(), Linear(6, 3))
    other_criterion = CrossEntropyLoss()
    other_optimizer = SGD(other_network.parameters(), lr=0.02)
    other_model = factory.model(network=other_network, criterion=other_criterion, optimizer=other_optimizer, device='cpu')
    models.restore(other_model, experiment)
    assert allclose(other_model.network[0].weight, model.network[0].weight)
    assert other_model.network[1].__class__ == model.network[1].__class__
    assert other_model.criterion.__class__ == model.criterion.__class__
    assert other_model.optimizer.param_groups[0]['lr'] == 0.01  

def test_datasets():
    data = Data(device='cpu')
    loader = data.get('mnist', train=True, batch_size=32)
    for input, target in loader:
        assert input.shape == (32, 1, 28, 28)
        assert target.shape == (32,)
        break