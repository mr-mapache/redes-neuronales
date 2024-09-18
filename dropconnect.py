from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from models.classifiers import DCP as DCCLS

from src.adapters import register
from src.adapters.experiments import Experiments, Settings
from src.application.exceptions import ExperimentAlreadyExists
from src.application.commands import TrainOverEpochs, CreateExperiment
from src.application.handlers import handle_create_experiment, handle_training_over_epochs
from logging import basicConfig, INFO, getLogger

basicConfig(level=INFO)
logger = getLogger(__name__)

# MLP's
register('nn','classiffier-dcp-784-256-10-0.0-relu', lambda: DCCLS(784, 256, 10, 0.0, 'relu'))
register('nn','classiffier-dcp-784-512-10-0.0-relu', lambda: DCCLS(784, 512, 10, 0.0, 'relu'))

register('nn','classiffier-dcp-784-256-10-0.2-relu', lambda: DCCLS(784, 256, 10, 0.2, 'relu'))
register('nn','classiffier-dcp-784-512-10-0.2-relu', lambda: DCCLS(784, 512, 10, 0.2, 'relu'))

register('nn','classiffier-dcp-784-256-10-0.5-relu', lambda: DCCLS(784, 256, 10, 0.5, 'relu'))
register('nn','classiffier-dcp-784-512-10-0.5-relu', lambda: DCCLS(784, 512, 10, 0.5, 'relu'))

# Cross-Entropy Loss and Adam Optimizer
register('criterion', 'cross-entropy', lambda: CrossEntropyLoss())
register('optimizer', 'adam-0.001', lambda model: Adam(model.parameters(), lr=0.001))

from pymongo import MongoClient
mongo_client = MongoClient('mongodb://localhost:27017')
database = mongo_client['experiments']
directory = 'data/weights'
settings = Settings(device='cuda', workers=4, database=database, directory=directory)
experiments = Experiments(settings)

namemaspace = '005-mnist-bz=256'

commands = [
    CreateExperiment(name=f'{namemaspace}-dcp-256-0.0-relu', nn='classiffier-dcp-784-256-10-0.0-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-256-0.2-relu', nn='classiffier-dcp-784-256-10-0.2-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-256-0.5-relu', nn='classiffier-dcp-784-256-10-0.5-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-512-0.0-relu', nn='classiffier-dcp-784-512-10-0.0-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-512-0.2-relu', nn='classiffier-dcp-784-512-10-0.2-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-512-0.5-relu', nn='classiffier-dcp-784-512-10-0.5-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
]

try:
    for command in commands:
        handle_create_experiment(command, experiments)
except ExperimentAlreadyExists as e:
    logger.info(e)

commands = [
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-256-0.0-relu', epochs=100, dataset='mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-256-0.2-relu', epochs=100, dataset='mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-256-0.5-relu', epochs=100, dataset='mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-512-0.0-relu', epochs=100, dataset='mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-512-0.2-relu', epochs=100, dataset='mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-512-0.5-relu', epochs=100, dataset='mnist'),
]

for command in commands:
    handle_training_over_epochs(command, experiments)


# 003 had no drop on bias.
# 004 has drop on bias.
# 005 has no bias in inference and not drop on bias in training.


namemaspace = '005-fashion-mnist-bz=256'

commands = [
    CreateExperiment(name=f'{namemaspace}-dcp-256-0.0-relu', nn='classiffier-dcp-784-256-10-0.0-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-256-0.2-relu', nn='classiffier-dcp-784-256-10-0.2-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-256-0.5-relu', nn='classiffier-dcp-784-256-10-0.5-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-512-0.0-relu', nn='classiffier-dcp-784-512-10-0.0-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-512-0.2-relu', nn='classiffier-dcp-784-512-10-0.2-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-dcp-512-0.5-relu', nn='classiffier-dcp-784-512-10-0.5-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
]

try:
    for command in commands:
        handle_create_experiment(command, experiments)
except ExperimentAlreadyExists as e:
    logger.info(e)

commands = [
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-256-0.0-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-256-0.2-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-256-0.5-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-512-0.0-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-512-0.2-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-dcp-512-0.5-relu', epochs=100, dataset='fashion-mnist'),
]

for command in commands:
    handle_training_over_epochs(command, experiments)