import pytest
from typing import Generator
from queue import Queue

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from fastapi import status
from fastapi.testclient import TestClient

from models import MLPCLS
from src.adapters.experiments import Experiments, Settings
from src.application.endpoints import api, repository, bus
from src.application.messagebus import Messagebus
from src.adapters.models.repository import Models
from src.adapters.states import States

States.register('mlp-cls-784-128-10-0.5-relu', 'nn')
States.register('cross-entropy', 'criterion')
States.register('adam-0.001', 'optimizer')

Models.register_nn('mlp-cls-784-512-10-0.5-relu', lambda: MLPCLS(784, 512, 10, 0.5))
Models.register_criterion('cross-entropy', lambda: CrossEntropyLoss())
Models.register_optimizer('adam-0.001', lambda model: Adam(model.parameters(), lr=0.001))

@pytest.fixture
def settings(mongo_database, directory) -> Settings:
    return Settings(
        device='cpu',
        workers=None,
        database=mongo_database,
        directory=directory
    )

@pytest.fixture
def adapter(settings) -> Experiments:
    return Experiments(settings)

@pytest.fixture
def client(adapter):
    api.dependency_overrides[repository] = lambda: adapter
    return TestClient(api)

def test_experiments(client: TestClient):
    response = client.post('/experiments/', json={
        'name': 'test',
        'nn': 'nn',
        'criterion': 'criterion',
        'optimizer': 'optimizer',
        'batch_size': 32
    })

    assert response.status_code == 400
    
    response = client.post('/experiments/', json={
        'name': 'test',
        'nn': 'mlp-cls-784-128-10-0.5-relu',
        'criterion': 'cross-entropy',
        'optimizer': 'adam-0.001',
        'batch_size': 32
    })

    assert response.status_code == 200
    
    response = client.post('/experiments/', json={
        'name': 'test',
        'nn': 'mlp-cls-784-128-10-0.5-relu',
        'criterion': 'cross-entropy',
        'optimizer': 'adam-0.001',
        'batch_size': 32
    })

    assert response.status_code == status.HTTP_409_CONFLICT

    
    response = client.post('/experiments/', json={
        'name': 'test2',
        'nn': 'mlp-cls-784-128-10-0.5-relu',
        'criterion': 'cross-entropy',
        'optimizer': 'adam-0.001',
        'batch_size': 64
    })

    assert response.status_code == 200

    response = client.get('/experiments/')

    assert response.status_code == 200
    experiment_list = response.json()
    assert len(experiment_list) == 2
    response = client.get('/experiments/name?name=test')
    assert response.status_code == 200
    experiment = response.json()
    assert experiment['name'] == 'test'
    response = client.get('/experiments/name?name=test2')
    assert response.status_code == 200    