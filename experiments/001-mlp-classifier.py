from torch.nn import Sequential
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.nn import Flatten

from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

from ops.model import Model
from ops.services import train, test
from ops.bus import Publisher, Consumer
from ops.handlers import handle_classification_results
from models.ffn.classifiers import MLP

from logging import basicConfig, getLogger
from logging import INFO

basicConfig(level=INFO)
logger = getLogger('ops')

device = 'cuda'
module = Sequential(Flatten(), MLP(784, 128, 10))
criterion = CrossEntropyLoss()
optimizer = Adam(module.parameters(), lr=0.001)
model = Model(module, criterion, optimizer).to(device)

publisher = Publisher()
publisher.subscribe('train-results', Consumer(handler=handle_classification_results))
publisher.subscribe('test-results', Consumer(handler=handle_classification_results))
publisher.subscribe('model-summary', Consumer(handler=logger.info))

datasets = {
    'train': MNIST(root='data', train=True, transform=ToTensor(), download=True), 
    'test': MNIST(root='data', train=False, transform=ToTensor(), download=True)
}

dataloaders = {
    'train': DataLoader(datasets['train'], batch_size=64, shuffle=True, pin_memory=True, pin_memory_device=device, num_workers=4),
    'test': DataLoader(datasets['test'], batch_size=64, shuffle=False, pin_memory=True, pin_memory_device=device, num_workers=4)
}

for epoch in range(4):
    train(model, dataloaders['train'], publisher, device)
    test(model, dataloaders['test'], publisher, device)

logger.log(INFO, 'Training and testing complete')

publisher.stop()