from torch.nn import Sequential
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import Flatten

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from models.ffn.classifiers import MLP
from logging import basicConfig, getLogger
from logging import INFO

from ops.publisher import Publisher
from ops.consumers import Metrics
from ops.repository import Models


from ops.model import Model
from ops.repository import Models
from ops.services import train, test


def run_experiment(model: Model, repository: Models, publisher: Publisher, dataloaders: dict[str, DataLoader], device: str):
    logger = getLogger(__name__)
    running = True
    try:
        logger.log(INFO, 'Restoring model')
        repository.restore(model)

        while running:
            try:
                train(model, dataloaders['train'], publisher, device)
                test(model, dataloaders['test'], publisher, device)
                logger.log(INFO, f'Epoch {model.epochs} complete')

                if model.epochs % 5 == 0:
                    logger.log(INFO, 'Saving model')
                    repository.save(model)
                model.epochs += 1

            except Exception:
                model = repository.create(model.name, model.network, model.criterion, model.optimizer, device)
                repository.restore(model)
                pass

    except KeyboardInterrupt:
        logger.log(INFO, 'Training stopped')
        repository.save(model)
                
    finally:
        publisher.stop()
    

basicConfig(level=INFO)
logger = getLogger('ops')

device = 'cuda'
network = Sequential(Flatten(), MLP(784, 128, 10))
criterion = CrossEntropyLoss()
optimizer = Adam(network.parameters(), lr=0.001)

publisher = Publisher()
publisher.subscribe('train-results', Metrics(device='cpu'))
publisher.subscribe('test-results', Metrics(device='cpu'))
models = Models('weights', device, publisher)
model = models.create('mlp', network, criterion, optimizer)

datasets = {
    'train': MNIST(root='data', train=True, transform=ToTensor(), download=True), 
    'test': MNIST(root='data', train=False, transform=ToTensor(), download=True)
}

dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=64, shuffle=True, pin_memory=True, pin_memory_device=device, num_workers=4),
    'test': DataLoader(datasets['test'], batch_size=64, shuffle=False, pin_memory=True, pin_memory_device=device, num_workers=4)
}


run_experiment(model, models, publisher, dataloaders, device)
