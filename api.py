from torch.nn import CrossEntropyLoss
from torch.optim import Adam

from models.classifiers import MLP as MLPCLS
from models.classifiers import GLU as GLUCLS

from src.adapters import register
from src.adapters.experiments import Experiments, Settings
from src.application.endpoints import api, repository, bus
from src.application.messagebus import Messagebus
from logging import basicConfig, INFO, getLogger

basicConfig(level=INFO)
logger = getLogger(__name__)

if __name__ == '__main__':
    from uvicorn import run
    from pymongo import MongoClient
    mongo_client = MongoClient('mongodb://localhost:27017')
    database = mongo_client['experiments']
    directory = 'data/weights'
    settings = Settings(device='cuda', workers=4, database=database, directory=directory)
    experiments = Experiments(settings)
    api.dependency_overrides[repository] = lambda: experiments
    run(api, host='0.0.0.0', port=8000)