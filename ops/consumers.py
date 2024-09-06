from queue import Queue
from threading import Thread
from logging import getLogger
import time

from ops.ports import Consumer
from ops.services import accuracy, predictions, Result

class Metrics(Consumer, Thread):
    def __init__(self, device: str) -> None:
        super().__init__(daemon=True)
        self.queue = Queue[Result]()
        self.device = device
        self.epoch = 1
        self.batch = 1
        self.loss = []
        self.accuracy = []
        self.average_loss = 0
        self.average_accuracy = 0
        self.logger = getLogger(__name__)
        self.start()

    def consume(self, message):
        self.queue.put(message)

    def calculate(self, result):
        if isinstance(result, int):
            self.epoch = result
            self.logger.info(f'Epoch {self.epoch} - Loss: {self.average_loss:.4f} - Accuracy: {self.average_accuracy:.4f}')
            self.loss.append(self.average_loss)
            self.accuracy.append(self.average_accuracy)
            self.average_loss = 0
            self.average_accuracy = 0
            self.batch = 1
        elif isinstance(result, Result):
            self.average_accuracy = (self.average_accuracy * self.batch + accuracy(predictions(result.output), result.target)) / (self.batch + 1)
            self.average_loss = (self.average_loss * self.batch + result.loss) / (self.batch + 1)
            self.batch += 1
        
    def run(self):
        while True:
            result = self.queue.get()
            if result is None:
                break
            self.calculate(result)
            self.queue.task_done()       

    def stop(self) -> None:
        self.consume(None)
        self.logger.info('Metrics stopped')
        self.join()