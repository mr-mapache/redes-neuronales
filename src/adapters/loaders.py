from typing import Callable
from typing import override
from typing import Optional
from logging import getLogger

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST, CIFAR10
from torchvision.transforms import ToTensor, Compose, Normalize

from src.domain.ports import Loaders as Repository
from src.domain.models import Loader

logger = getLogger(__name__)

class SimpleMnist(Dataset):
    def __init__(self, train: bool) -> None:
        self.transform = ToTensor()
        self.dataset = MNIST(root='./data/datasets', train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

class NormalizedMnist(Dataset):
    def __init__(self, train: bool) -> None:
        self.transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
        self.dataset = MNIST(root='./data/datasets', train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]


class Cifar10(Dataset):
    def __init__(self, train: bool) -> None:
        self.transform = ToTensor()
        self.dataset = CIFAR10(root='./data/datasets', train=train, download=True, transform=self.transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index] 

class Datasets:
    registry: dict[str, Callable[[bool], Dataset]] = {
        'mnist': lambda train: SimpleMnist(train=train),
        'fashion-mnist': lambda train: FashionMNIST(train=train)
    }

    @classmethod
    def register(cls, name: str, factory: Callable[[bool, Callable], Dataset]):
        cls.registry[name] = factory

    @classmethod
    def get(cls, dataset: str, train: bool) -> Optional[Dataset]:
        factory = cls.registry.get(dataset)
        if not factory:
            raise ValueError(f'Unknown dataset: {dataset}')
        return factory(train)
    

class Loaders(Repository):
    datasets = Datasets

    def __init__(self, device: str, workers: Optional[int]):
        self.device = device
        if self.device == 'cpu':
            if workers:
                raise ValueError('Workers are not supported on CPU')
        self.workers = workers or 0

    @override
    def get(self, dataset: str, train: bool, batch_size: int) -> Loader:
        dataset = Datasets.get(dataset, train)
        if self.device == 'cpu':
            return DataLoader(dataset, batch_size=batch_size, shuffle=train)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=train, pin_memory=True, pin_memory_device=self.device, num_workers=self.workers)