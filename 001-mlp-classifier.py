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

basicConfig(level=INFO)
logger = getLogger('ops')

device = 'cuda'
network = Sequential(Flatten(), MLP(784, 128, 10))
criterion = CrossEntropyLoss()
optimizer = Adam(network.parameters(), lr=0.001)

publisher = Publisher()
publisher.subscribe('train-results', Metrics(device='cpu'))
publisher.subscribe('test-results', Metrics(device='cpu'))

from ops.repository import Models

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

from ops.services import train, test

for epoch in range(1, 5):
    train(epoch, model, dataloaders['train'], publisher, device)
    test(epoch, model, dataloaders['test'], publisher, device)
    logger.log(INFO, f'Epoch {model.epochs} complete')
    model.epochs += 1

logger.log(INFO, 'Training and testing complete')
logger.log(INFO, 'Saving model')
models.save(model)


model = models.create('mlp', network, criterion, optimizer)
models.restore(model)

for epoch in range(10, 14):
    train(epoch, model, dataloaders['train'], publisher, device)
    test(epoch, model, dataloaders['test'], publisher, device)
    logger.log(INFO, f'Epoch {model.epochs} complete')
    model.epochs += 1

logger.log(INFO, 'Training and testing complete')

publisher.stop()