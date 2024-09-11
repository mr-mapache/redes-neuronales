from torch import Tensor
from src.adapters.metrics.values import Loss, Accuracy, predictions
from src.domain.models import Phase

def test_accuracy():
    accuracy = Accuracy()
    output = Tensor([
        [0.1, 0.2, 0.7],
        [0.3, 0.2, 0.5],
        [0.1, 0.1, 0.8],
        [0.1, 0.2, 0.7]
    ])
    target = Tensor([1, 2, 1 ,2])
    assert accuracy(1, output, target, 'train') == 0.5
    assert accuracy(2, output, target, 'train') == 0.5
    accuracy(0, None, None, Phase.BREAK)
    assert accuracy.average == 0.0
    assert accuracy.history[Phase.TRAIN] == [0.5]
    assert accuracy(1, output, target, 'evaluation') == 0.5
    assert accuracy(2, output, target, 'evaluation') == 0.5
    assert accuracy(3, output, target, 'evaluation') == 0.5
    accuracy(0, None, None, Phase.BREAK)
    assert accuracy.history[Phase.TRAIN] == [0.5]
    assert accuracy.history[Phase.EVALUATION] == [0.5]
    assert accuracy.average == 0.0
    target = Tensor([1, 2, 1 ,1])
    assert accuracy(1, output, target, 'train') == 0.25
    assert accuracy(2, output, target, 'train') == 0.25
    
def test_loss():
    loss = Loss()
    assert loss(1, 0.5, 'train') == 0.5
    assert loss(2, 0.3, 'train') == 0.4
    loss(0, None, Phase.BREAK)
