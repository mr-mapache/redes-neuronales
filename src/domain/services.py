from typing import Callable
from typing import Tuple
from typing import Literal
from src.domain.models import Model, Loader, Result, Callback, Phase

def train(model: Model, loader: Loader, callback: Callback, device: str):
    model.train()
    for batch, (input, target) in enumerate(loader, start=1):
        input, target = input.to(device), target.to(device)
        output, loss = model.fit(input, target)
        callback(Result(batch, input, target, output, loss, Phase.TRAIN))
    callback(Result(0, None, None, None, 0.0, Phase.BREAK))

def evaluate(model: Model, loader: Loader, callback: Callback, device: str):
    model.eval()
    for batch, (input, target) in enumerate(loader, start=1):
        input, target = input.to(device), target.to(device)
        output, loss = model.evaluate(input, target)
        callback(Result(batch, input, target, output, loss, Phase.EVALUATION))
    callback(Result(0, None, None, None, 0.0, Phase.BREAK))