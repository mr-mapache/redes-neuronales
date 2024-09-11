from torch.optim import Adam
from torch.nn import CrossEntropyLoss

from models.classifiers import MLP as MLPCLS
from src.adapters.states import States
from src.adapters.models.factory import Builder

States.register('mlp-cls-784-128-10-0.5-relu', 'classifier')
States.register('cross-entropy', 'criterion')
States.register('adam-0.001', 'optimizer')

Builder.register_nn('mlp-cls-784-128-10-0.5-relu', lambda: MLPCLS(784, 128, 10, 0.5))
Builder.register_criterion('cross-entropy', CrossEntropyLoss)
Builder.register_optimizer('adam-0.001', lambda model: Adam(model.parameters(), lr=0.001))