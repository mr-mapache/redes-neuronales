from typing import Callable
from typing import override
from ops.ports import Data as DAO
from ops.models import Loader
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import Compose, ToTensor

def get(dataset: str, train: bool, transform: Callable | None = None) -> Dataset:
    transform = transform or ToTensor()
    dataset = dataset.lower()
    match dataset:
        case 'mnist':
            return MNIST(root='./data/datasets', train=train, download=True, transform=transform)
        case 'fashionmnist':
            return FashionMNIST(root='./data/datasets', train=train, download=True, transform=transform)
        case _:
            raise ValueError(f'Unknown dataset: {dataset}')

class Data(DAO):
    def __init__(self, device: str = 'cuda', workers: int = 4):
        self.device = device
        self.workers = workers
        
    @override
    def get(self, dataset: str, train: bool, batch_size: int, transform: Callable | None = None) -> Loader:
        dataset = get(dataset, train, transform)
        if self.device == 'cpu':
            return DataLoader(dataset, batch_size=batch_size, shuffle=train)
        else:
            return DataLoader(dataset, batch_size=batch_size, shuffle=train, pin_memory=True, num_workers=self.workers)