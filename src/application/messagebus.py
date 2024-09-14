from typing import Callable
from threading import Thread
from queue import Queue, Empty
from src.application.commands import Command, TrainOverEpochs
from src.application.handlers import handle_training_over_epochs


class Messagebus(Thread):
    def __init__(self, experiments):
        super().__init__()
        raise Exception('This class is deprecated, starting a new thread slows down the application')
        self.experiments = experiments
        self.handlers: dict[type[Command], Callable[[Command], None]] = {
            TrainOverEpochs: lambda command: handle_training_over_epochs(command, self.experiments)
        }
        self.queue = Queue()
        self.start()

    def enqueue(self, command: Command):
        raise Exception('This class is deprecated, starting a new thread slows down the application')
        self.queue.put(command)

    def handle(self, command: Command):
        raise Exception('This class is deprecated, starting a new thread slows down the application')
        handler = self.handlers.get(type(command), None)
        if handler:
            handler(command)

    def run(self):
        raise Exception('This class is deprecated, starting a new thread slows down the application')
        while True:
            try:
                command = self.queue.get()
                if command is None:
                    break
                self.handle(command)
                self.queue.task_done()
            except Empty:
                continue

    def stop(self):
        raise Exception('This class is deprecated, starting a new thread slows down the application')
        self.queue.put(None)
        self.join()