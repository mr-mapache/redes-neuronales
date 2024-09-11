from typing import Callable
from typing import override
from typing import Optional
from logging import getLogger

from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor

from src.domain.ports import Loaders as Repository
from src.domain.models import Loader

logger = getLogger(__name__)

class Datasets:
    registry: dict[str, Callable[[bool, Callable], Dataset]] = {
        'mnist': lambda train, transform: MNIST(root='./data/datasets', train=train, download=True, transform=transform),
        'fashion-mnist': lambda train, transform: FashionMNIST(root='./data/datasets', train=train, download=True, transform=transform)
    }

    @classmethod
    def register(cls, name: str, factory: Callable[[bool, Callable], Dataset]):
        cls.registry[name] = factory

    @classmethod
    def get(cls, dataset: str, train: bool, transform: Callable) -> Optional[Dataset]:
        factory = cls.registry.get(dataset)
        if not factory:
            raise ValueError(f'Unknown dataset: {dataset}')
        return factory(train, transform)
    

class Loaders(Repository):
    datasets = Datasets

    def __init__(self, device: str, workers: Optional[int]):
        self.device = device
        if self.device == 'cpu':
            if workers:
                raise ValueError('Workers are not supported on CPU')
        self.workers = workers or 0

    @override
    def get(self, dataset: str, train: bool, batch_size: int, transform: Callable | None = None) -> Loader:
        transform = transform or ToTensor()
        dataset = Datasets.get(dataset, train, transform)
        if self.device == 'cpu':
            return DataLoader(dataset, batch_size=batch_size, shuffle=train)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=train, pin_memory=True, pin_memory_device=self.device, num_workers=self.workers)