from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from models.classifiers import MLP as MLPCLS
from models.classifiers import GLU as GLUCLS

from src.adapters import register
from src.adapters.experiments import Experiments, Settings
from src.application.exceptions import ExperimentAlreadyExists
from src.application.commands import TrainOverEpochs, CreateExperiment
from src.application.handlers import handle_create_experiment, handle_training_over_epochs
from logging import basicConfig, INFO, getLogger

basicConfig(level=INFO)
logger = getLogger(__name__)

# MLP's
register('nn','classiffier-mlp-784-256-10-0.0-relu', lambda: MLPCLS(784, 256, 10, 0.0))
register('nn','classiffier-mlp-784-512-10-0.0-relu', lambda: MLPCLS(784, 512, 10, 0.0))

register('nn','classiffier-mlp-784-256-10-0.2-relu', lambda: MLPCLS(784, 256, 10, 0.2))
register('nn','classiffier-mlp-784-512-10-0.2-relu', lambda: MLPCLS(784, 512, 10, 0.2))

register('nn','classiffier-mlp-784-256-10-0.5-relu', lambda: MLPCLS(784, 256, 10, 0.5))
register('nn','classiffier-mlp-784-512-10-0.5-relu', lambda: MLPCLS(784, 512, 10, 0.5))

# GLU's
register('nn','classiffier-glu-784-256-10-0.0-relu', lambda: GLUCLS(784, 256, 10, 0.0))
register('nn','classiffier-glu-784-512-10-0.0-relu', lambda: GLUCLS(784, 512, 10, 0.0))

register('nn','classiffier-glu-784-256-10-0.2-relu', lambda: GLUCLS(784, 256, 10, 0.2))
register('nn','classiffier-glu-784-512-10-0.2-relu', lambda: GLUCLS(784, 512, 10, 0.2))

register('nn','classiffier-glu-784-256-10-0.5-relu', lambda: GLUCLS(784, 256, 10, 0.5))
register('nn','classiffier-glu-784-512-10-0.5-relu', lambda: GLUCLS(784, 512, 10, 0.5))

# Cross-Entropy Loss and Adam Optimizer
register('criterion', 'cross-entropy', lambda: CrossEntropyLoss())
register('optimizer', 'adam-0.001', lambda model: Adam(model.parameters(), lr=0.001))

from pymongo import MongoClient
mongo_client = MongoClient('mongodb://localhost:27017')
database = mongo_client['experiments']
directory = 'data/weights'
settings = Settings(device='cuda', workers=4, database=database, directory=directory)
experiments = Experiments(settings)

namemaspace = '001-fashion-mnist-bz=256'

commands = [
    CreateExperiment(name=f'{namemaspace}-mlp-256-0.0-relu', nn='classiffier-mlp-784-256-10-0.0-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-mlp-256-0.2-relu', nn='classiffier-mlp-784-256-10-0.2-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-mlp-256-0.5-relu', nn='classiffier-mlp-784-256-10-0.5-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-mlp-512-0.0-relu', nn='classiffier-mlp-784-512-10-0.0-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-mlp-512-0.2-relu', nn='classiffier-mlp-784-512-10-0.2-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-mlp-512-0.5-relu', nn='classiffier-mlp-784-512-10-0.5-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),

    CreateExperiment(name=f'{namemaspace}-glu-256-0.0-relu', nn='classiffier-glu-784-256-10-0.0-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-glu-256-0.2-relu', nn='classiffier-glu-784-256-10-0.2-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-glu-256-0.5-relu', nn='classiffier-glu-784-256-10-0.5-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-glu-512-0.0-relu', nn='classiffier-glu-784-512-10-0.0-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
    CreateExperiment(name=f'{namemaspace}-glu-512-0.2-relu', nn='classiffier-glu-784-512-10-0.2-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256), 
    CreateExperiment(name=f'{namemaspace}-glu-512-0.5-relu', nn='classiffier-glu-784-512-10-0.5-relu', criterion='cross-entropy', optimizer='adam-0.001', batch_size=256),
]

try:
    for command in commands:
        handle_create_experiment(command, experiments)
except ExperimentAlreadyExists as e:
    logger.info(e)

commands = [
    TrainOverEpochs(experiment=f'{namemaspace}-mlp-256-0.0-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-mlp-256-0.2-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-mlp-256-0.5-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-mlp-512-0.0-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-mlp-512-0.2-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-mlp-512-0.5-relu', epochs=100, dataset='fashion-mnist'),

    TrainOverEpochs(experiment=f'{namemaspace}-glu-256-0.0-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-glu-256-0.2-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-glu-256-0.5-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-glu-512-0.0-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-glu-512-0.2-relu', epochs=100, dataset='fashion-mnist'),
    TrainOverEpochs(experiment=f'{namemaspace}-glu-512-0.5-relu', epochs=100, dataset='fashion-mnist'),
]

for command in commands:
    handle_training_over_epochs(command, experiments)