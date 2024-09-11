from pymongo import MongoClient

from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torchvision.transforms import Compose, ToTensor, Normalize

from models.classifiers import MLP as MLPCLS
from models.classifiers import GLU as GLUCLS
from src.adapters.experiments import Experiments, Settings
from src.application.bus import MessageBus
from src.application.commands import CreateExperiment, TrainOverEpochs
from logging import basicConfig, INFO, getLogger

MessageBus.register('nn','mlp-cls-784-128-10-0.5-relu', lambda: MLPCLS(784, 128, 10, 0.5))
MessageBus.register('nn','mlp-cls-784-256-10-0.5-relu', lambda: MLPCLS(784, 256, 10, 0.5))
MessageBus.register('nn','mlp-cls-784-512-10-0.5-relu', lambda: MLPCLS(784, 512, 10, 0.5))

MessageBus.register('nn','glu-cls-784-128-10-0.5-relu', lambda: GLUCLS(784, 128, 10, 0.5))
MessageBus.register('nn','glu-cls-784-256-10-0.5-relu', lambda: GLUCLS(784, 256, 10, 0.5))
MessageBus.register('nn','glu-cls-784-512-10-0.5-relu', lambda: GLUCLS(784, 512, 10, 0.5))

MessageBus.register('criterion', 'cross-entropy', lambda: CrossEntropyLoss())
MessageBus.register('optimizer', 'adam-0.001', lambda model: Adam(model.parameters(), lr=0.001))


mongo_client = MongoClient('mongodb://localhost:27017')
database = mongo_client['runs']
directory = 'data/weights'

settings = Settings(
    device='cuda',
    workers=4,
    database=database,
    directory=directory
)

if __name__ == '__main__':
    basicConfig(level=INFO)
    logger = getLogger(__name__)

    experiments = Experiments(settings)

    bus = MessageBus(experiments)

    bus.handle(CreateExperiment(
        name='001-classifier-glu-128',
        nn='glu-cls-784-128-10-0.5-relu',
        criterion='cross-entropy',
        optimizer='adam-0.001',
        batch_size=64
    ))

    bus.handle(CreateExperiment(
        name='002-classifier-glu-256',
        nn='glu-cls-784-256-10-0.5-relu',
        criterion='cross-entropy',
        optimizer='adam-0.001',
        batch_size=64
    ))

    bus.handle(CreateExperiment(
        name='003-classifier-glu-512',
        nn='glu-cls-784-512-10-0.5-relu',
        criterion='cross-entropy',
        optimizer='adam-0.001',
        batch_size=64
    ))

    bus.handle(TrainOverEpochs(
        epochs=50,
        experiment='001-classifier-glu-128',
        dataset='mnist',
        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        task='classification'
    ))

    
    bus.handle(TrainOverEpochs(
        epochs=50,
        experiment='002-classifier-glu-256',
        dataset='mnist',
        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        task='classification'
    ))

    
    bus.handle(TrainOverEpochs(
        epochs=50,
        experiment='003-classifier-glu-512',
        dataset='mnist',
        transform=Compose([ToTensor(), Normalize((0.1307,), (0.3081,))]),
        task='classification'
    ))