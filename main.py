from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from models.classifiers import MLP as MLPCLS
from models.classifiers import GLU as GLUCLS

from src.adapters import register
from src.adapters.experiments import Experiments, Settings
from src.application.endpoints import api, repository
from logging import basicConfig, INFO, getLogger

basicConfig(level=INFO)
logger = getLogger(__name__)

register('nn','mlp-cls-784-128-10-0.5-relu', lambda: MLPCLS(784, 128, 10, 0.5))
register('nn','mlp-cls-784-256-10-0.5-relu', lambda: MLPCLS(784, 256, 10, 0.5))
register('nn','mlp-cls-784-512-10-0.5-relu', lambda: MLPCLS(784, 512, 10, 0.5))

register('nn','glu-cls-784-128-10-0.5-relu', lambda: GLUCLS(784, 128, 10, 0.5))
register('nn','glu-cls-784-256-10-0.5-relu', lambda: GLUCLS(784, 256, 10, 0.5))
register('nn','glu-cls-784-512-10-0.5-relu', lambda: GLUCLS(784, 512, 10, 0.5))

register('criterion', 'cross-entropy', lambda: CrossEntropyLoss())
register('optimizer', 'adam-0.001', lambda model: Adam(model.parameters(), lr=0.001))

if __name__ == '__main__':
    from uvicorn import run
    from pymongo import MongoClient
    mongo_client = MongoClient('mongodb://localhost:27017')
    database = mongo_client['runs']
    directory = 'data/weights'
    settings = Settings(device='cuda', workers=4, database=database, directory=directory)
    experiments = Experiments(settings)
    api.dependency_overrides[repository] = lambda: experiments
    run(api, host='0.0.0.0', port=8000)